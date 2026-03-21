# GalaxyCSM

`GalaxyCSM`은 은하 화학 진화(Galactic Chemical Evolution, GCE) 결과를 바탕으로 별, 항성 진화, 행성계, 대기, 내부 구조, 위성계, 소행성대까지 생성하고 웹에서 시각화하는 Flask 기반 시뮬레이터입니다.

브라우저에서 은하 지도를 탐색하면서 개별 항성계의 진화 이력과 행성 물리 정보를 함께 볼 수 있도록 구성되어 있습니다.

## 주요 기능

- 다중 반경 구역 기반 은하 화학 진화 계산
- 별 질량, 나이, 금속성에 따른 항성 진화 및 H-R 트랙 시각화
- 행성계 생성 및 행성 물성 계산
- 대기, 광분해, 내부 구조, 자기장, 거주가능성 관련 파생 값 계산
- 거대행성 및 해왕성형 행성의 위성계 생성
- 원시 행성원반과 소행성대 정보 표시
- Flask API + 정적 프론트엔드 기반 웹 인터페이스

## 기술 스택

- Python
- Flask
- NumPy
- SciPy
- Matplotlib
- Three.js
- Plotly

## 빠른 시작

### 1. 환경 준비

```powershell
cd "c:\Users\soomi\Downloads\Projects\GalaxyCSM\GalaxyCSM-main"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 서버 실행

```powershell
python server.py
```

실행 후 브라우저에서 아래 주소로 접속합니다.

- `http://127.0.0.1:5000/`

## 사용 방법

1. 브라우저에서 시뮬레이션을 실행합니다.
2. 은하 지도에서 별을 선택해 항성계 상세 정보를 확인합니다.
3. H-R 다이어그램과 GCE 차트를 열어 진화 경향을 비교합니다.
4. 필터와 파라미터 창에서 분광형, 진화 단계, 행성 조건, GCE 계수를 조절합니다.

## API 개요

서버는 정적 UI와 함께 몇 가지 핵심 API를 제공합니다.

항성 진화 엔진은 `stellar_model`로 선택할 수 있습니다.

- `auto`: 사전계산 트랙이 있으면 정밀 보간, 아니면 휴리스틱 fallback
- `precise`: 정밀 트랙 보간 우선 사용
- `heuristic`: 기존 휴리스틱 트랙만 사용

### `GET /api/defaults`

기본 시뮬레이션 파라미터와 기본 별 개수를 반환합니다.

### `POST /api/galaxy`

은하 시뮬레이션을 실행하고 별 목록 및 GCE 결과를 반환합니다.

예시:

```bash
curl -X POST http://127.0.0.1:5000/api/galaxy ^
  -H "Content-Type: application/json" ^
  -d "{\"n_stars\": 5000, \"t_max\": 20.0, \"stellar_model\": \"auto\"}"
```

주요 입력값 예시:

- `n_stars`: 생성할 별 개수
- `t_max`: GCE 계산 시간 범위(Gyr)
- `sfr_efficiency`
- `outflow_eta`
- `yield_s_multiplier`
- `yield_r_multiplier`
- `yield_ia_multiplier`
- `agb_frequency_multiplier`
- `stellar_model`

### `GET /api/star/<star_id>`

특정 별의 현재 진화 상태와 행성계 상세 정보를 반환합니다.

예시:

```text
GET /api/star/5?t=13.8&stellar_model=auto&cache_id=<cache_id>
```

### `GET /api/evolution/<star_id>`

특정 별의 H-R 진화 트랙을 반환합니다.

예시:

```text
GET /api/evolution/5?t_max=100.0&stellar_model=auto&cache_id=<cache_id>
```

## 테스트

이 프로젝트의 테스트는 `unittest` 기반 API 테스트와 직접 실행형 검증 스크립트가 함께 들어 있습니다.

### API 테스트

아래 명령은 실제로 통과 확인한 기본 테스트 명령입니다.

```powershell
python -m unittest test_galaxy.py test_modes.py
```

### 추가 검증 스크립트

필요에 따라 개별 스크립트를 직접 실행할 수 있습니다.

```powershell
python test_physical_consistency.py
python test_moons.py
python test_planets.py
python test_disk.py
python test_precise_tracks.py
python benchmark_stellar_tracks.py
```

참고:

- 일부 파일은 자동 수집형 테스트라기보다 진단/검증 스크립트에 가깝습니다.
- 성능 측정이나 디버깅 목적의 테스트 파일도 포함되어 있습니다.

## 프로젝트 구조

```text
GalaxyCSM-main/
|- server.py                  # Flask 앱 및 API 엔드포인트
|- requirements.txt           # Python 의존성
|- static/                    # 웹 UI
|  |- index.html
|  |- css/
|  `- js/
|- gce/                       # 시뮬레이션 핵심 모듈
|  |- config.py               # 기본 상수와 파라미터
|  |- solver.py               # GCE 솔버
|  |- stellar.py              # 별 생성 및 항성 진화
|  |- stellar_tracks.py       # H-R 진화 트랙
|  |- stellar_remnants.py     # 백색왜성/중성자별/블랙홀 처리
|  |- planet_generation.py    # 행성계 생성
|  |- planets.py              # 행성 물리 및 거주가능성 계산
|  |- planet_atmosphere.py    # 대기 모델
|  |- planet_interior.py      # 내부 구조/열 진화
|  |- photolysis.py           # 광분해 관련 계산
|  |- disk.py                 # 원시 행성원반/소행성대
|  `- moons.py                # 위성계 생성
`- test_*.py                  # 테스트 및 검증 스크립트
```

## 구현 포인트

- `gce/config.py`에는 추적 원소, 태양 조성, 기본 시뮬레이션 파라미터가 정의되어 있습니다.
- `gce/solver.py`는 반경과 시간 축에서 원소 진화를 계산하는 핵심 솔버입니다.
- `gce/stellar.py`와 관련 모듈은 별 생성, 항성 진화 단계, 현재 상태, H-R 트랙 계산을 담당합니다.
- `gce/planets.py` 중심의 행성 모듈은 생성된 항성계에 대해 행성 물리, 대기, 내부 구조, 위성계를 확장 계산합니다.

## 주의 사항

- 기본 별 개수는 비교적 큰 편이므로 실행 환경에 따라 초기 시뮬레이션이 오래 걸릴 수 있습니다.
- 빠르게 확인하려면 API 호출 시 `n_stars`를 작은 값으로 시작하는 것이 좋습니다.
- 이 프로젝트는 패키지 설치형보다는 `server.py`를 직접 실행하는 구조입니다.

## 라이선스

저장소에 별도 라이선스 파일이 보이지 않아 현재는 라이선스 정보가 명시되어 있지 않습니다. 배포 또는 외부 공유 전에는 라이선스 정책을 먼저 확인하는 것을 권장합니다.
