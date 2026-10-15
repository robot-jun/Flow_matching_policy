"""Visualize held-out demonstrations vs conditional Flow Matching predictions.
Run from project root: python visualize.py
"""
# 이 스크립트는 모델을 학습하지 않고, 저장된 모델의 예측 결과를 시각화합니다.
# 순서: 모델/데이터 로딩 → 작업별 경유점 예측 → 오차 계산 → PNG/HTML/JSON 저장.
# 좌표 계산은 미터(m), 그래프 및 저장 결과는 밀리미터(mm)를 사용합니다.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
# 화면 창 없이 파일로 그림을 저장하는 Agg 백엔드입니다. 서버 환경에서도 사용 가능합니다.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# make_subplots는 현재 코드에서는 사용하지 않는 import입니다.
from plotly.subplots import make_subplots
# 학습 코드에서 모델 구조, 조건 벡터 생성 함수, 경유점 생성 함수를 가져옵니다.
from waypoint_fm import Flow,condition,sample_waypoints,DEFAULT_OUTPUT

# 현재 파이썬 파일이 있는 폴더를 기준 경로로 사용합니다.
# 모델과 평가 예제는 학습 코드와 동일하게 2번 폴더의 output에서 읽습니다.
# 시각화 결과는 3번 폴더에 저장하며 터미널의 실행 위치와 무관합니다.
# matplotlib와 plotly도 별도로 설치되어 있어야 합니다.
ROOT=Path(__file__).resolve().parent
MODEL_OUT=DEFAULT_OUTPUT
# 결과 저장 폴더를 만듭니다. parents=True는 없는 상위 폴더도 만들고 exist_ok=True는 기존 폴더를 허용합니다.
OUT=ROOT.parent/'FM_03_Visualizations.'/'output'/'visualizations';OUT.mkdir(exist_ok=True,parents=True)
# 체크포인트에는 학습 가중치, 작업 목록, 중간점 수 k, 정규화 정보가 있습니다. CPU로 불러옵니다.
# weights_only=False로 읽으므로 신뢰할 수 있는 모델 파일을 사용해야 합니다.
ckpt=torch.load(MODEL_OUT/'model.pt',map_location='cpu',weights_only=False)
# 학습 당시와 같은 모델 구조를 복원합니다. 출력은 XYZ × k개(기본 12차원).
# 조건은 시작 XYZ 3개 + 목표-시작 변위 3개 + 작업 종류 one-hot 벡터입니다.
model=Flow(3*ckpt['k'],6+len(ckpt['tasks']),ckpt['width'])
# 학습 가중치를 적용하고 평가 모드로 전환합니다. 가중치를 갱신하는 학습 과정은 없습니다.
model.load_state_dict(ckpt['model']);model.eval()
# 학습에서 제외한 CSV로 만든 평가 예제 목록을 읽습니다.
examples=json.loads((MODEL_OUT/'test_examples.json').read_text())
# 평가 예제들이 같은 시연 CSV에서 왔다고 가정하여 첫 예제의 원본 CSV를 읽습니다.
raw=pd.read_csv(ROOT/'data'/examples[0]['csv'])
# 작업 이름이 직전 행과 달라질 때마다 번호를 증가시켜 연속 작업 구간을 구분합니다.
groups=raw.current_task.ne(raw.current_task.shift()).cumsum()
# 작업 이름으로 원본 구간을 찾는 사전입니다. 같은 이름이 여러 구간에 반복되면 마지막 구간만 남습니다.
by_task={str(g.current_task.iloc[0]):g for _,g in raw.groupby(groups,sort=False)}
# results는 JSON 평가 결과, traces는 종합 그래프에서 재사용할 경로 좌표를 저장합니다.
results=[];traces=[]
# 평가 예제를 순서대로 처리합니다. i는 0부터 시작하는 번호이고 e는 한 작업 구간입니다.
for i,e in enumerate(examples):
    # start/goal은 (3,), ref는 (k, 3) 배열이며 단위는 m입니다.
    # ref는 원본 전체 궤적이 아니라 표시된 waypoint와 양 끝점을 시간 인덱스로 보간한 정답입니다.
    start=np.array(e['start_m']);goal=np.array(e['goal_m']); ref=np.array(e['target_m'])
    # 작업/시작/목표를 조건으로 Flow Matching 모델이 내부 경유점 k개를 생성합니다.
    # condition은 조건 벡터를 만들고, cond_mean/std는 입력 정규화에 사용됩니다.
    # yscale은 모델이 생성한 직선 대비 잔차를 원래 스케일로 복원합니다.
    # 80은 Euler 적분 횟수, 42는 초기 노이즈 시드, cpu는 추론 장치입니다.
    # sample_waypoints 내부에서는 기울기 계산을 끄고 노이즈를 점진적으로 변환합니다.
    pred=sample_waypoints(model,condition(e,ckpt['tasks']),start,goal,np.asarray(ckpt['cond_mean']),np.asarray(ckpt['cond_std']),np.asarray(ckpt['yscale']),ckpt['k'],80,42,'cpu')
    # 비교 기준인 직선의 내부 점입니다. k=4이면 시작~목표의 20/40/60/80% 위치입니다.
    # [1:-1]은 양 끝점을 제외하고, None은 (k, 1) 차원을 만들어 XYZ 연산을 가능하게 합니다.
    straight=start+(goal-start)*np.linspace(0,1,ckpt['k']+2)[1:-1,None]
    # 원본 CSV의 말단 장치(TCP) XYZ 전체 기록을 가져와 mm에서 m로 변환합니다.
    rawpath=by_task[e['task']][['EE_x','EE_y','EE_z']].to_numpy(float)/1000
    # rawpath에는 해당 작업 구간에서 기록된 모든 TCP 위치 샘플이 들어 있습니다.
    # 그래프용 좌표는 mm로 변환합니다. [None]은 (3,) 좌표를 (1, 3) 행으로 만듭니다.
    # np.r_는 시작점 + 내부 점들 + 목표점을 행 방향으로 연결합니다.
    # mp=모델 예측, rr=보간 정답, sg=직선 기준, truepath=원본 기록 궤적입니다.
    mp=np.r_[start[None],pred,goal[None]]*1000
    rr=np.r_[start[None],ref,goal[None]]*1000
    sg=np.r_[start[None],straight,goal[None]]*1000
    truepath=rawpath*1000
    # 각 내부 점의 XYZ 유클리드 거리를 구하고 k개 평균을 계산해 mm로 바꿉니다.
    # axis=1은 각 점의 XYZ 방향입니다. baseerr는 같은 방식으로 구한 직선 기준 오차입니다.
    # 실제 궤적 전체와의 거리나 작업 성공률, 충돌 여부를 측정하는 지표는 아닙니다.
    err=np.linalg.norm(pred-ref,axis=1).mean()*1000
    baseerr=np.linalg.norm(straight-ref,axis=1).mean()*1000
    # NumPy 배열을 JSON으로 저장 가능한 리스트로 바꿉니다. 오차는 소수 2자리, 좌표는 3자리로 반올림합니다.
    result={'task':e['task'],'model_error_mm':round(float(err),2),'straight_baseline_error_mm':round(float(baseerr),2),'raw_waypoint_marks':e['raw_mark_count'],'start_mm':(start*1000).tolist(),'goal_mm':(goal*1000).tolist(),'generated_waypoints_mm':(pred*1000).round(3).tolist()}
    results.append(result)
    # 작업 이름과 네 종류 경로를 보관하여 아래 종합 그래프에서 다시 사용합니다.
    traces.append((e['task'],truepath,mp,rr,sg))
    # 작업별 3D 그림을 만듭니다. 회색=실제 기록, 주황=예측, 청록 점선=보간 정답.
    # 좌표 배열의 .T는 X/Y/Z를 행으로 바꾸고, *는 세 배열을 plot 인수로 펼칩니다.
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
    ax.plot(*truepath.T,color='#263238',lw=1.5,alpha=.65,label='Recorded TCP trajectory')
    ax.plot(*mp.T,color='#e4572e',lw=2.2,marker='o',markersize=5,label='Flow Matching + endpoints')
    ax.plot(*rr.T,color='#2a9d8f',lw=1.7,ls='--',marker='x',label='Interpolated supervision')
    # 시작점은 녹색, 목표점은 파란색으로 표시합니다. 모델이 생성하는 것은 그 사이의 점들입니다.
    ax.scatter(*start*1000,color='green',s=70,label='Start');ax.scatter(*goal*1000,color='blue',s=70,label='Goal')
    ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)');ax.set_title(f'{e["task"]}\nFM {err:.1f} mm | straight baseline {baseerr:.1f} mm',fontsize=11);ax.legend(fontsize=7,loc='upper left')
    # 모든 표시 좌표의 최솟값/최댓값으로 범위를 계산합니다. span은 최소 100mm로 잡습니다.
    # mid는 범위 중심이며, 아래에서 세 축에 같은 수치 폭과 여유 공간을 적용합니다.
    a=np.vstack([truepath,mp,rr]);span=np.maximum(a.max(axis=0)-a.min(axis=0),100);mid=(a.max(axis=0)+a.min(axis=0))/2
    ax.set_xlim(mid[0]-span.max()*.55,mid[0]+span.max()*.55);ax.set_ylim(mid[1]-span.max()*.55,mid[1]+span.max()*.55);ax.set_zlim(mid[2]-span.max()*.55,mid[2]+span.max()*.55)
    # 레이아웃을 정리하고 번호와 작업 이름이 붙은 PNG로 저장한 후 메모리의 그림을 닫습니다.
    fig.tight_layout();fig.savefig(OUT/f'{i+1:02d}_{e["task"]}.png',dpi=155);plt.close(fig)

# 전체 작업을 X-Z 평면에서 비교하는 요약 그림입니다.
# 전체 작업을 4×4 패널에 X-Z 평면으로 투영합니다. Y축 정보는 이 그림에서 생략됩니다.
# 현재 배치는 최대 16개 구간을 가정합니다. axs.flat은 패널을 1차원 순서로 접근합니다.
fig,axs=plt.subplots(4,4,figsize=(18,14),constrained_layout=True)
for i,(task,rawpath,mp,rr,sg) in enumerate(traces):
    ax=axs.flat[i];ax.plot(rawpath[:,0],rawpath[:,2],color='#555',label='Recorded TCP',lw=1.3);ax.plot(mp[:,0],mp[:,2],'-o',color='#e4572e',label='FM',ms=3);ax.plot(rr[:,0],rr[:,2],'--x',color='#2a9d8f',label='Target',ms=3);ax.plot(sg[:,0],sg[:,2],':',color='#78909c',label='Straight',lw=1)
    ax.set_title(task.replace('_',' '),fontsize=9);ax.set_xlabel('X (mm)',fontsize=8);ax.set_ylabel('Z (mm)',fontsize=8);ax.tick_params(labelsize=7);ax.grid(alpha=.25)
# 작업 수보다 남는 패널은 숨깁니다. 다음 줄에서는 첫 패널의 범례를 공통으로 사용합니다.
for j in range(len(traces),16):axs.flat[j].axis('off')
handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower right',bbox_to_anchor=(.96,.09),ncol=2)
fig.suptitle('Held-out demonstration (CSV #3) — XYZ waypoint predictions, XZ view',fontsize=16)
fig.savefig(OUT/'all_segments_xz.png',dpi=170);plt.close(fig)

# 작업별 FM 오차와 직선 오차를 막대그래프로 비교합니다.
# xx는 작업 위치, ±0.2는 두 막대의 좌우 이동, 0.4는 각 막대 폭입니다.
fig,ax=plt.subplots(figsize=(12,6));names=[r['task'].replace('_',' ') for r in results];xx=np.arange(len(names));ax.bar(xx-.2,[r['model_error_mm'] for r in results],.4,label='Flow Matching');ax.bar(xx+.2,[r['straight_baseline_error_mm'] for r in results],.4,label='Straight-line baseline');ax.set_xticks(xx,names,rotation=50,ha='right');ax.set_ylabel('Mean target waypoint distance (mm)');ax.set_title('Held-out CSV: Flow Matching vs straight line');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'error_comparison.png',dpi=170);plt.close(fig)

# Interactive 3D all traces with menu showing one task at a time
# 마우스로 회전/확대할 수 있는 Plotly 3D 그래프입니다.
# 작업 하나당 경로 4개 + 시작/목표 1개 = trace 5개입니다. trace는 독립적인 표시 요소입니다.
fig=go.Figure();ntr=5
for i,(task,rawpath,mp,rr,sg) in enumerate(traces):
    for vals,name,color,dash,markers in [(rawpath,'Recorded TCP','#555555','solid',False),(mp,'Flow Matching','#e4572e','solid',True),(rr,'Interpolated target','#2a9d8f','dash',True),(sg,'Straight baseline','#7d8b99','dot',False)]:
        # 좌표 배열의 0/1/2열을 X/Y/Z로 사용합니다. markers 설정에 따라 점도 표시합니다.
        # visible=i==3은 네 번째 작업만 초기 화면에 표시한다는 뜻입니다.
        fig.add_trace(go.Scatter3d(x=vals[:,0],y=vals[:,1],z=vals[:,2],mode='lines+markers' if markers else 'lines',name=name,visible=i==3,line=dict(color=color,width=7 if name=='Flow Matching' else 4,dash=dash),marker=dict(size=4)))
    # 시작과 목표에 색상 점과 텍스트를 추가합니다. 이 요소도 해당 작업의 trace에 포함됩니다.
    fig.add_trace(go.Scatter3d(x=[mp[0,0],mp[-1,0]],y=[mp[0,1],mp[-1,1]],z=[mp[0,2],mp[-1,2]],mode='markers+text',text=['Start','Goal'],textposition='top center',marker=dict(color=['green','blue'],size=6),name='Start/Goal',visible=i==3))
# 드롭다운에서 작업을 선택할 때 해당 작업의 trace 다섯 개만 표시하도록 설정합니다.
buttons=[]
for i,(task,*_) in enumerate(traces):
    # 전체 trace 번호 j를 ntr(5)로 나눈 몫이 작업 번호입니다. 선택한 작업 i만 True가 됩니다.
    visible=[j//ntr==i for j in range(len(traces)*ntr)]
    # 버튼 선택 시 표시 여부 배열과 그래프 제목을 함께 갱신합니다.
    buttons.append(dict(label=task,method='update',args=[{'visible':visible},{'title':f'Flow Matching vs held-out CSV: {task}'}]))
# 축 단위는 mm이고 aspectmode=data는 실제 데이터 범위에 비례해 축을 표시합니다.
# 초기 제목은 네 번째 작업을 사용하므로 평가 구간이 최소 4개 있어야 합니다.
fig.update_layout(title=f'Flow Matching vs held-out CSV: {traces[3][0]}',scene=dict(xaxis_title='X (mm)',yaxis_title='Y (mm)',zaxis_title='Z (mm)',aspectmode='data'),updatemenus=[dict(buttons=buttons,x=0,y=1.12,xanchor='left',yanchor='top')],margin=dict(t=110,l=0,r=0,b=0),legend=dict(orientation='h',y=-.05))
# Plotly 자바스크립트까지 HTML에 포함하여 별도 CDN 연결 없이 브라우저에서 열 수 있습니다.
fig.write_html(OUT/'interactive_3d.html',include_plotlyjs=True)
# 구간별 수치와 전체 평균을 JSON으로 저장합니다.
# 반올림된 구간 오차를 평균하므로 report.json의 정밀한 평균과 끝자리가 조금 다를 수 있습니다.
# 이 스크립트의 그림과 오차는 XYZ 예측 평가이며 IK나 충돌 검사를 수행하지 않습니다.
(OUT/'visual_evaluation.json').write_text(json.dumps({'heldout_csv':examples[0]['csv'],'results':results,'mean_fm_mm':float(np.mean([r['model_error_mm'] for r in results])),'mean_straight_baseline_mm':float(np.mean([r['straight_baseline_error_mm'] for r in results]))},indent=2))
# 저장 위치와 모델/직선 기준의 전체 평균 오차를 터미널에 출력합니다.
print('Saved',OUT,'FM',np.mean([r['model_error_mm'] for r in results]),'straight',np.mean([r['straight_baseline_error_mm'] for r in results]))
