# Flow_matching_policy

(EN)
The overall goal of folders FM_01–FM_03 is to learn from robot demonstrations and automatically generate intermediate waypoints between a given start and goal for a specific task.
For example, given the task “move from the feeder retreat position to the machine approach position,” the model proposes four intermediate XYZ waypoints based on the demonstrated motion patterns.

① FM_01_Code_Data — Data preparation and model training
This folder contains the demonstration data and Python code.
The code divides each demonstration into task segments and creates four target waypoints by interpolating the marked waypoints together with the segment’s start and end positions. It then trains a conditional Flow Matching model:
Task type + start XYZ + goal XYZ → four intermediate XYZ waypoints

- waypoint_fm.py: Prepares data, trains the model, and generates waypoints.
- visualize.py: Compares predictions with reference paths.
- CSV files: Recorded robot demonstrations.
- URDF file: Robot description; the current code only reads and stores joint limits.
  
② FM_02_Model_Results — Trained model and evaluation results
This folder stores the trained model, evaluation examples, generated predictions, and evaluation report.
The model was trained on 26 motion segments from two demonstrations and evaluated on 13 segments from the remaining demonstration.
The saved results show a mean waypoint error of approximately 61.23 mm. This measures the distance between predicted waypoints and interpolated reference waypoints.

③ FM_03_Visualizations — Visual inspection of predictions
This folder contains plots and an interactive 3D visualization comparing:
- The recorded robot trajectory
- Model-generated waypoints
- Interpolated reference waypoints
- A straight-line baseline between the start and goal

These comparisons help identify whether the model reproduces demonstrated path shapes, which tasks have large errors, and whether predictions improve on the straight-line baseline.
The complete workflow is:
Robot demonstrations → reference waypoint preparation → Flow Matching training → waypoint generation → evaluation and visualization
The current implementation is a research prototype for generating and evaluating XYZ waypoint proposals. Camera-based perception, robot orientation generation, inverse kinematics, collision checking, and actual robot control are outside its implemented scope.


(KR)
이 1~3번 폴더가 궁극적으로 하려는 일은 사람이 시연한 로봇 이동 데이터를 학습해서, 특정 작업의 시작점과 목표점 사이에 필요한 중간 경유점을 자동으로 생성하는 것입니다.
예를 들어, 로봇에 다음 조건을 주는 거예요.
“공급기에서 후퇴한 위치에서 출발해 가공기 접근 위치까지 이동해야 한다.”

그러면 모델이 시연에서 배운 경로 형태를 바탕으로 중간에 지나갈 XYZ 좌표 4개를 제안합니다.

① FM_01_Code_Data — 시연 데이터를 학습하는 단계
CSV 3개에 기록된 로봇 작업을 작업 이름별로 나눕니다. 각 구간에서 시작점·목표점·waypoint 표시를 이용해 학습용 중간점 4개를 만듭니다.
이후 작업 종류와 시작·목표 좌표가 주어졌을 때, 어떤 경유점을 생성해야 하는지를 조건부 Flow Matching으로 학습합니다.
- waypoint_fm.py: 데이터 처리, 학습, 경유점 예측
- visualize.py: 예측 결과를 그래프로 비교
- CSV: 시연 데이터
- URDF: 로봇 정보. 현재는 관절 제한값을 읽어 보관하는 용도

② FM_02_Model_Results — 학습 결과를 저장하고 평가하는 단계
학습한 모델과 예측 결과, 평가 성적표를 보관합니다.
시연 2개에서 얻은 26개 구간으로 학습하고, 남은 시연 1개의 13개 구간으로 평가했습니다. 저장된 결과에서 평균 경유점 오차는 약 61.23mm입니다.
이 수치는 모델이 만든 중간점과 시연의 waypoint 표시를 보간해서 만든 정답 사이의 거리입니다.

③ FM_03_Visualizations — 배운 경로가 적절한지 확인하는 단계
다음 경로들을 그림으로 비교합니다.
- 실제 CSV에 기록된 로봇 궤적
- 모델이 생성한 경유점
- 학습·평가에 사용한 보간 정답
- 시작과 목표를 바로 연결한 직선

이를 통해 모델이 시연의 경로 형태를 얼마나 따라 하는지, 어느 작업에서 오차가 큰지, 직선보다 나은지를 확인합니다.
전체 흐름은 이렇게 연결됩니다.
로봇 시연 CSV → 작업별 정답 경유점 생성 → Flow Matching 학습 → 새로운 조건의 경유점 생성 → 오차 평가·시각화
현재 프로젝트의 완성 범위는 **“로봇 이동에 사용할 XYZ 경유점을 제안하고 평가하는 연구용 모델”**입니다. 카메라로 상황을 인식하거나, 관절 명령을 생성하거나, 충돌을 피하며 로봇을 실제로 제어하는 단계까지 구현된 것은 아닙니다.
