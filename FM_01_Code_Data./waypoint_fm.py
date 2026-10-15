"""Machine-tending sparse XYZ waypoint conditional flow matching.
Uses only uploaded demonstration CSVs and URDF joint-limit metadata.
CSV EE position is in mm, URDF length in meters. Predictions are NOT robot-safe by default.
"""
# 이 파일은 조건부 Flow Matching으로 로봇의 중간 XYZ 경유점을 생성합니다.
# 입력: 작업 종류 + 시작 XYZ + 목표 XYZ / 출력: 중간 XYZ k개(기본 4개).
# 처리 흐름: CSV 구간 분리 → 정답 보간 → 직선 대비 잔차 학습 → 잔차 생성 → XYZ 복원.
# CSV/사용자 입출력은 mm, 모델 내부 좌표는 m입니다.
# 관절 명령이나 말단 자세를 생성하지 않으며, 역기구학 및 충돌 검사는 구현되어 있지 않습니다.
import argparse, glob, json, random, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

# CSV에서 사용할 말단 장치 위치 열과 6개 관절각 열 이름입니다.
# 관절각은 존재 여부와 유효 숫자인지만 확인하며 모델 입력으로 사용하지 않습니다.
XYZ = ['EE_x','EE_y','EE_z']
JOINTS = [f'joint{i}_angle' for i in range(1,7)]

# 실행한 터미널 위치와 무관하게 프로젝트의 2번 폴더에 학습 결과를 저장합니다.
# --out을 직접 지정하면 사용자가 지정한 경로를 우선합니다.
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / 'FM_02_Model_Results.' / 'output'

# Python, NumPy, PyTorch 난수 시드를 고정해 같은 환경에서 실행 결과의 재현성을 높입니다.
def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

# URDF XML에서 joint1~joint6의 관절 제한값을 읽어 메타데이터로 보관합니다.
# 제한값을 학습 손실이나 예측 제약에 적용하지는 않습니다. tcp_link도 여기서는 고정 문자열입니다.
def read_urdf(path):
    root=ET.parse(path).getroot();limits={}
    for j in root.findall('joint'):
        if j.attrib.get('name') in [f'joint{i}' for i in range(1,7)]:
            x=j.find('limit');limits[j.attrib['name']]={'lower_rad':float(x.attrib['lower']),'upper_rad':float(x.attrib['upper'])}
    # 대상 관절 6개가 모두 발견되지 않으면 잘못된 로봇 파일로 판단합니다.
    if len(limits)!=6:raise ValueError('Expected all six robot revolute joints in URDF')
    return {'robot':root.attrib.get('name'), 'joint_limits':limits, 'tcp_link':'tool_center_point', 'collision_validation':'NOT IMPLEMENTED'}

# 시연 CSV를 읽어 고정 크기 학습 예제를 만듭니다. k는 내부 경유점 수입니다.
# 반환값: 예제 목록, 작업 이름 목록, URDF 메타데이터, CSV별 행 수 진단 정보.
def load_examples(folder, urdf, k=4):
    # 파일 이름을 정렬한 순서가 file_idx=0/1/2가 됩니다. 현재 구현은 CSV가 정확히 3개여야 합니다.
    files=sorted(glob.glob(str(Path(folder)/'*.csv')))
    if len(files)!=3:raise ValueError(f'Expected exactly 3 CSVs in {folder}, found {len(files)}')
    info=read_urdf(urdf);items=[];tasks=set();diagnostics=[]
    for fi,f in enumerate(files):
        d=pd.read_csv(f)
        # 필수 열과 숫자 유효성을 확인합니다. NaN 또는 무한대가 있으면 중단합니다.
        for x in XYZ+['current_task','waypoint']+JOINTS:
            if x not in d:raise ValueError(f'Missing {x} in {f}')
        if not np.isfinite(d[XYZ+JOINTS].to_numpy(float)).all():raise ValueError(f'NaN / inf in positions/joints: {f}')
        # current_task가 직전 행과 달라질 때마다 번호를 올려 연속된 작업 구간을 분리합니다.
        groups=d.current_task.ne(d.current_task.shift()).cumsum()
        for _,g in d.groupby(groups,sort=False):
            task=str(g.current_task.iloc[0]);tasks.add(task)
            # 행이 너무 적은 구간은 제외합니다. 시작/끝은 해당 구간의 첫 행과 마지막 행입니다.
            if len(g)<3:continue
            # 원본 mm 좌표를 m로 변환합니다. 시작~끝 거리가 15mm 미만인 구간은 제외합니다.
            # 전체 이동 길이가 아니라 두 끝점 사이 거리 기준이므로 돌아오는 경로도 제외될 수 있습니다.
            start=g[XYZ].iloc[0].to_numpy(float)/1000.;end=g[XYZ].iloc[-1].to_numpy(float)/1000.
            if np.linalg.norm(end-start)<0.015:continue  # stationary operation, not transit
            # Marked points plus end anchors, interpolate at equally-spaced normalized indices.
            # waypoint 열의 공백과 대소문자를 정리하고 값이 waypoint인 행의 구간 내부 위치를 찾습니다.
            hits=np.flatnonzero(g.waypoint.astype(str).str.strip().str.lower().eq('waypoint').to_numpy())
            if len(hits)==0:continue # unmarked segments cannot supervise sparse landmarks
            # 표시된 점과 시작/끝 행 번호를 합친 뒤 중복을 제거합니다.
            # 표시가 양 끝점에만 있다면 보간 정답은 시작~끝 직선이 될 수 있습니다.
            x_idx=np.unique(np.r_[0,hits,len(g)-1]);xy=g[XYZ].iloc[x_idx].to_numpy(float)/1000.
            # 행 번호를 0~1로 정규화합니다. 실제 타임스탬프나 이동 거리 기준은 아닙니다.
            # sample은 내부 k개 위치입니다. k=4이면 0.2, 0.4, 0.6, 0.8입니다.
            t=x_idx/(len(g)-1); sample=np.linspace(0,1,k+2)[1:-1]
            # X/Y/Z를 각각 선형 보간해 (k, 3) 정답을 만듭니다.
            # 원본 전체 궤적을 정답으로 쓰는 것이 아니라 표시된 점들 사이를 보간합니다.
            target=np.column_stack([np.interp(sample,t,xy[:,i]) for i in range(3)])
            # Task identity, current start joint angles (URDF degrees vs radians unclear):
            # use XYZ start / goal, and task only, to avoid implicit unit assumptions.
            # 시작과 끝을 직선으로 연결한 기준 좌표입니다. [None, :]는 XYZ를 한 행으로 만듭니다.
            # 모델은 절대 좌표 대신 정답이 이 직선에서 얼마나 벗어났는지(잔차)를 학습합니다.
            base=start[None,:]+sample[:,None]*(end-start)[None,:]
            # (k, 3) 잔차를 길이 3*k 벡터로 펼칩니다. k=4이면 12차원입니다.
            residual=(target-base).reshape(-1)
            # 파일/작업 식별 정보, m 단위 좌표, 정답 잔차 및 원래 표시 개수를 저장합니다.
            items.append({'file_idx':fi,'csv':Path(f).name,'task':task,'start_m':start.tolist(),
                          'goal_m':end.tolist(),'target_m':target.tolist(),'residual_m':residual.tolist(),
                          'raw_mark_count':int(len(hits)), 'rows':int(len(g))})
        diagnostics.append({'csv':Path(f).name,'rows':len(d)})
    # 학습에 사용할 수 있는 구간에서 작업 이름을 모아 정렬합니다.
# 작업 목록은 전체 CSV에서 만들지만 가중치와 정규화 통계는 아래에서 학습 CSV만 사용합니다.
    task_names=sorted({i['task'] for i in items})
    if not items:raise ValueError('No nonstationary, waypoint-marked segments to learn')
    return items,task_names,info,diagnostics

# 한 예제의 조건 벡터를 만듭니다. 길이는 6 + 작업 종류 수입니다.
# 절대 시작 위치와 상대 목표 변위, 작업 종류를 함께 전달합니다.
def condition(e,task_names):
    # normalize all geometry to meters and centered on start; absolute start retained.
    a=np.asarray(e['start_m'],dtype=np.float32);b=np.asarray(e['goal_m'],dtype=np.float32)
    # one-hot은 해당 작업 위치만 1이고 다른 위치는 0인 범주 표현입니다.
    hot=np.zeros(len(task_names),np.float32);hot[task_names.index(e['task'])]=1
    # 시작 XYZ + 목표-시작 XYZ + one-hot을 한 벡터로 연결합니다.
    return np.r_[a,b-a,hot].astype(np.float32)

# 조건부 속도장 v(x, t, c)를 예측하는 다층 퍼셉트론입니다.
# x는 현재 잔차 공간 상태, t는 생성 과정의 시간, c는 작업/좌표 조건입니다.
# 여기서 속도는 로봇의 실제 이동 속도가 아니라 잔차 생성 공간의 변화율입니다.
class Flow(nn.Module):
    def __init__(self,dim,cond_dim,width=128):
        # 입력 차원: 잔차 dim + 조건 cond_dim + 시간 1. 은닉층은 SiLU 활성화를 사용합니다.
        # 출력 차원은 dim으로, 현재 상태를 어느 방향으로 바꿀지 예측합니다.
        super().__init__();self.net=nn.Sequential(nn.Linear(dim+cond_dim+1,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,dim))
    def forward(self,x,t,c):
        # 배치 차원은 유지하고 마지막 특징 축으로 x, t, c를 연결합니다.
        return self.net(torch.cat([x,t,c],dim=-1))

# 지정된 CSV 번호의 예제만 선택하고 조건 C와 잔차 정답 Y를 배열로 만듭니다.
# 예제 N개일 때 C는 (N, 6+작업 수), Y는 (N, 3*k)입니다.
def make_arrays(items,task_names,ids):
    selected=[e for e in items if e['file_idx'] in ids]
    return selected,np.stack([condition(e,task_names) for e in selected]),np.asarray([e['residual_m'] for e in selected],np.float32)

# 학습 모드: 파일 단위로 학습/평가를 분리하고 모델·평가 예제·성적표를 저장합니다.
def train(args):
    set_seed(args.seed);device=torch.device(args.device)
    items,tasks,urdf,diag=load_examples(args.data_dir,args.urdf,args.k)
    # test_csv로 고른 1개 CSV를 제외한 2개만 학습에 사용합니다. 같은 시연의 행을 무작위 분할하지 않습니다.
    train_examples,C,Y=make_arrays(items,tasks,[i for i in range(3) if i!=args.test_csv])
    # 남겨 둔 CSV로 평가 예제를 만듭니다. Ct/Yt는 이 코드에서 이후 직접 사용하지 않습니다.
    test_examples,Ct,Yt=make_arrays(items,tasks,[args.test_csv])
    # Task generalization is not evaluated: if a task occurs only in held-out demonstration, skip eval that task.
    # 학습에 등장하지 않은 작업 종류는 평가에서 제외합니다. 새로운 작업 종류에 대한 일반화 평가는 아닙니다.
    known={e['task'] for e in train_examples}; eval_examples=[e for e in test_examples if e['task'] in known]
    if len(eval_examples)!=len(test_examples):print('Warning: skipped unseen task(s) in evaluation')
    # 학습 데이터만으로 조건의 평균/표준편차를 구합니다. 표준편차 하한은 0.05입니다.
    # 거의 변하지 않는 특징을 나눌 때 값이 지나치게 커지는 것을 줄입니다.
    cond_mean=C.mean(0);cond_std=C.std(0);cond_std=np.maximum(cond_std,0.05)
    # Do not center one-hot features: preserve explicit categorical features.
    # 작업 one-hot은 정규화하지 않고 원래의 0/1 값을 유지합니다.
    cond_mean[6:]=0.;cond_std[6:]=1.
    # 잔차 차원마다 학습 표준편차로 크기를 맞추며 최소 스케일은 0.10m입니다.
    # 잔차 평균을 빼지는 않습니다. 추론 시 동일 스케일을 곱해 원래 단위를 복원합니다.
    yscale=np.maximum(Y.std(0),0.10)  # per-position residual scaling, in meters
    yn=Y/yscale[None,:];cn=(C-cond_mean)/cond_std
    net=Flow(Y.shape[1],C.shape[1],args.width).to(device)
    # AdamW 최적화기로 가중치를 갱신합니다. lr은 학습률, weight_decay는 가중치 감쇠입니다.
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    x=torch.tensor(yn,device=device);c=torch.tensor(cn,device=device)
    # steps는 전체 데이터 순회 횟수(epoch)가 아니라 파라미터 갱신 횟수입니다.
    for step in range(1,args.steps+1):
        # 학습 예제 번호를 복원 추출해 미니배치를 구성합니다. 같은 예제가 중복 선택될 수 있습니다.
        ix=torch.randint(len(x),(args.batch,),device=device)
        # x1은 정규화된 정답 잔차, z는 같은 모양의 표준정규 노이즈입니다.
        # t는 예제마다 0~1 사이에서 뽑습니다. 실제 로봇의 실행 시간이 아닙니다.
        x1=x[ix];z=torch.randn_like(x1);t=torch.rand(len(ix),1,device=device)
        # Flow Matching 핵심: 노이즈 z와 정답 x1 사이의 직선 보간 상태 xt를 만듭니다.
        # xt=(1-t)*z+t*x1의 t에 대한 미분은 x1-z이며, 이것이 정답 속도입니다.
        xt=(1-t)*z+t*x1;velocity=x1-z
        # 예측 속도와 정답 속도의 평균제곱오차(MSE)를 줄입니다.
        # 이전 기울기 초기화 → 역전파 → 가중치 갱신 순서로 학습합니다.
        loss=((net(xt,t,c[ix])-velocity)**2).mean();opt.zero_grad();loss.backward();opt.step()
        if step==1 or step%max(1,args.steps//5)==0:print(f'step {step:5d}/{args.steps} train FM loss={loss.item():.5f}')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    path=out/'model.pt'
    # 추론 때 같은 모델과 좌표 변환을 복원할 수 있도록 가중치와 설정을 함께 저장합니다.
    torch.save({'model':net.state_dict(),'tasks':tasks,'k':args.k,'width':args.width,'cond_mean':cond_mean.tolist(),
                'cond_std':cond_std.tolist(),'yscale':yscale.tolist(),'urdf':urdf,'train_csv_ids':[i for i in range(3) if i!=args.test_csv],
                'test_csv_id':args.test_csv,'seed':args.seed},path)
    # 성적표에는 데이터 개수와 작업 목록, 평가 해석 시 필요한 한계를 기록합니다.
    report={'data_diagnostics':diag,'train_segments':len(train_examples),'test_segments':len(test_examples),
            'evaluated_segments':len(eval_examples),'task_names':tasks,'train_raw_mark_counts': [e['raw_mark_count'] for e in train_examples],
            'notes':['K internal points are interpolated from sparse CSV waypoint flags, not direct ground-truth collision-free targets.',
                     'Evaluation is held-out demonstration prediction vs interpolated labels, not collision/success rate.',
                     'Object/grasp pose is assumed encoded indirectly by start/goal and task; no object/grasp pose columns exist in CSV.',
                     'URDF supplies robot joint limits, but does not establish CSV-to-URDF EE frame alignment. No IK or collision check implemented.']}
    # 학습에서 제외한 시연의 각 작업에 대해 경유점을 생성하고 정답과 비교합니다.
    if eval_examples:
        errors=[]
        for e in eval_examples:
            pred=sample_waypoints(net,condition(e,tasks),e['start_m'],e['goal_m'],cond_mean,cond_std,yscale,args.k,args.ode_steps,args.seed,device)
            # 점별 3차원 유클리드 거리를 평균한 뒤 m→mm로 변환합니다.
            # 충돌률이나 전체 기록 궤적 오차가 아니라 보간 정답 중간점과의 거리입니다.
            errors.append(float(np.linalg.norm(pred-np.array(e['target_m']),axis=1).mean()*1000))
        report['heldout_mean_waypoint_error_mm']=float(np.mean(errors))
        report['heldout_per_segment_error_mm']=[{'task':e['task'],'error_mm':round(er,2)} for e,er in zip(eval_examples,errors)]
    # 성적표와 평가 예제를 JSON으로 저장합니다. 학습 모델은 앞서 model.pt에 저장했습니다.
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (out/'test_examples.json').write_text(json.dumps(test_examples,indent=2,ensure_ascii=False))
    print('Model saved:',path,'\nReport:',out/'report.json');print('Heldout mean waypoint error [mm]:',report.get('heldout_mean_waypoint_error_mm'))

# 추론 공통 함수: 노이즈를 학습된 잔차로 바꾸고 직선 기준에 더해 (k, 3) XYZ를 반환합니다.
# 입력/출력 좌표는 m이며, 시작점과 목표점은 반환 배열에 포함하지 않습니다.
def sample_waypoints(net,raw_cond,start,goal,mean,std,yscale,k,steps,seed,device):
    # Explicit Euler integration of learned conditional velocity field t=0 .. t=1.
    # 지역 난수 생성기를 사용합니다. 같은 시드/조건/모델은 같은 환경에서 같은 초기 노이즈를 만듭니다.
    generator=torch.Generator(device=device).manual_seed(int(seed))
    # 배치 1개, 차원 3*k의 표준정규 노이즈에서 생성을 시작합니다.
    z=torch.randn((1,3*k),device=device,generator=generator)
    # 학습 시 저장한 평균/표준편차로 조건을 정규화하고 배치 차원을 추가합니다.
    cc=torch.tensor(((raw_cond-mean)/std)[None,:],dtype=torch.float32,device=device)
    net.eval()
    # 추론에서는 역전파에 필요한 기울기를 저장하지 않습니다.
    with torch.no_grad():
        # 생성 시간 t=0부터 1까지 고정 간격 dt=1/steps로 진행합니다.
        for i in range(steps):
            tt=torch.full((1,1),i/steps,device=device)
            # 명시적 Euler 적분: 현재 상태에 예측 속도 × dt를 더합니다.
            # 입력 조건은 고정하고 잔차 상태 z만 반복적으로 갱신합니다.
            z=z+net(z,tt,cc)/steps
    # 최종 상태를 (k, 3)으로 바꾸고 정규화 스케일을 되돌려 m 단위 잔차를 얻습니다.
    delta=z.cpu().numpy().reshape(k,3)*yscale.reshape(k,3)
    # 입력 시작/목표 사이의 직선 내부 k개 점을 다시 계산합니다.
    base=np.asarray(start)[None,:]+np.linspace(0,1,k+2)[1:-1,None]*(np.asarray(goal)-np.asarray(start))[None,:]
    # 직선 기준 + 생성된 잔차가 최종 중간 경유점입니다. 이후 별도의 경로 검증은 수행하지 않습니다.
    return base+delta

# 사용자가 지정한 작업 이름과 시작/목표(mm)로 경유점을 생성하는 명령입니다.
def predict(args):
    device=torch.device(args.device)
    # 체크포인트는 신뢰할 수 있는 파일을 사용해야 합니다. weights_only=False로 전체 정보를 읽습니다.
    ckpt=torch.load(args.model,map_location=device,weights_only=False)
    # 저장된 설정을 읽고 사용자 입력 mm를 모델의 m 단위로 변환합니다.
    k=ckpt['k'];tasks=ckpt['tasks'];start=np.asarray(args.start_mm,float)/1000;goal=np.asarray(args.goal_mm,float)/1000
    # 학습 당시 등록된 작업 이름만 허용합니다.
    if args.task not in tasks:raise ValueError(f'Task not in trained labels: {tasks}')
    # 저장된 차원/은닉층 크기로 네트워크를 만든 뒤 학습 가중치를 복원합니다.
    model=Flow(3*k,6+len(tasks),ckpt['width']).to(device);model.load_state_dict(ckpt['model'])
    e={'start_m':start,'goal_m':goal,'task':args.task}
    pred=sample_waypoints(model,condition(e,tasks),start,goal,np.asarray(ckpt['cond_mean']),
                          np.asarray(ckpt['cond_std']),np.asarray(ckpt['yscale']),k,args.ode_steps,args.seed,device)*1000
    # 예측 좌표를 mm로 출력합니다. 경고는 좌표계·역기구학·충돌 검증이 필요함을 알립니다.
    result={'task':args.task,'start_xyz_mm':list(np.round(start*1000,3)),'goal_xyz_mm':list(np.round(goal*1000,3)),
            'intermediate_waypoints_xyz_mm':pred.round(3).tolist(),'warning':'Unvalidated XYZ proposals. Check TCP frame, IK, and full-path collisions before robot execution.'}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False))

# 평가 예제에서 해당 작업의 시작/목표를 찾아 predict를 재사용하는 편의 명령입니다.
def demo(args):
    # 같은 작업 이름이 여러 개 있으면 첫 번째 일치 예제를 선택합니다.
    samples=json.loads(Path(args.samples).read_text());e=next((x for x in samples if x['task']==args.task),None)
    if e is None:raise ValueError(f'No held-out example for task {args.task}')
    import types
    # 원래 명령행 인자를 복사하고 평가 예제의 시작/목표를 mm로 변환해 추가합니다.
    a=types.SimpleNamespace(**vars(args));a.start_mm=(np.array(e['start_m'])*1000).tolist();a.goal_mm=(np.array(e['goal_m'])*1000).tolist()
    predict(a)

# 명령행 진입점입니다. train(학습), predict(직접 예측), demo(평가 예제 예측) 중 하나를 선택합니다.
# 상대 경로 인자는 현재 터미널 작업 폴더를 기준으로 해석됩니다.
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    # 학습 인자: CSV 폴더, URDF 파일, 결과 폴더를 지정합니다.
    tr=sub.add_parser('train');tr.add_argument('--data-dir',default='data');tr.add_argument('--urdf',required=True);tr.add_argument('--out',default=str(DEFAULT_OUTPUT))
    # k는 내부 점 개수이고 test-csv는 정렬된 CSV 중 평가용 번호입니다(기본 2 → 세 번째 파일).
    tr.add_argument('--k',type=int,default=4);tr.add_argument('--test-csv',type=int,choices=[0,1,2],default=2)
    # steps는 가중치 갱신 횟수, batch는 한 번에 뽑는 예제 수, lr은 학습률입니다.
    tr.add_argument('--steps',type=int,default=1500);tr.add_argument('--batch',type=int,default=32);tr.add_argument('--lr',type=float,default=0.001)
    # width는 신경망 은닉 차원이며 ode-steps는 평가/생성 시 적분 횟수입니다.
    tr.add_argument('--width',type=int,default=128);tr.add_argument('--ode-steps',type=int,default=80)
    tr.add_argument('--seed',type=int,default=42);tr.add_argument('--device',default='cpu')
    # 직접 예측에는 저장된 모델 경로, 등록된 작업 이름, 시작/목표 XYZ가 필요합니다.
    pr=sub.add_parser('predict');pr.add_argument('--model',required=True);pr.add_argument('--task',required=True)
    # nargs=3이므로 좌표마다 X Y Z 세 숫자를 mm 단위로 입력합니다.
    pr.add_argument('--start-mm',type=float,nargs=3,required=True);pr.add_argument('--goal-mm',type=float,nargs=3,required=True)
    pr.add_argument('--ode-steps',type=int,default=80);pr.add_argument('--seed',type=int,default=42);pr.add_argument('--device',default='cpu');pr.add_argument('--output')
    # 데모는 모델과 평가 예제 JSON을 읽고 지정한 작업의 좌표로 예측합니다.
    dm=sub.add_parser('demo');dm.add_argument('--model',required=True);dm.add_argument('--samples',required=True)
    dm.add_argument('--task',default='retreat_feeder_to_approach_machine');dm.add_argument('--ode-steps',type=int,default=80)
    dm.add_argument('--seed',type=int,default=42);dm.add_argument('--device',default='cpu');dm.add_argument('--output')
    # 명령행 인자를 해석하고 선택한 모드에 해당하는 함수를 호출합니다.
    args=p.parse_args();{'train':train,'predict':predict,'demo':demo}[args.mode](args)
# 직접 실행할 때만 main을 호출합니다. visualize.py에서 import할 때는 학습/예측 명령이 실행되지 않습니다.
if __name__=='__main__':main()
