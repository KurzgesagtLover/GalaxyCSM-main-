# Stellar Track Pack Format

이 디렉터리는 사전계산된 항성 진화 트랙 팩을 저장합니다.

## 파일 형식

- 기본 포맷: `json`
- 스키마 버전: `1`
- 기본 파일명: `demo_precise_tracks.json`

## 최상위 필드

- `schema_version`: 현재 스키마 버전
- `coordinate_system`: 현재는 `phase_fraction_v1`
- `source`: 트랙 출처 설명
- `generated_by`: 생성 스크립트 경로
- `phase_order`: UI/API와 호환되는 phase 토큰 순서
- `tracks`: 개별 `(initial_mass, metallicity_z)` 트랙 배열

## 각 트랙의 필수 필드

- `initial_mass`
- `metallicity_z`
- `ages_gyr`
- `phases`
- `phase_fraction`
- `luminosity`
- `T_eff`
- `radius_rsun`
- `current_mass`
- `max_radius_rsun`
- `flare_activity`

## 보간 규약

- 나이 축은 절대 `age_gyr` 기준으로 검색합니다.
- 서로 다른 phase 경계에서는 선형 혼합 대신 가장 가까운 phase 점으로 스냅합니다.
- 같은 phase 안에서는 `luminosity`, `T_eff`, `radius_rsun`, `max_radius_rsun`를 로그 공간에서 보간합니다.
- 질량/금속성 축은 인접한 네 이웃 트랙의 가중 평균을 사용합니다.
- phase가 강하게 충돌하면 precise 엔진은 `None`을 반환하고 휴리스틱 엔진으로 fallback 합니다.
