# 다중 용어 역치환 샘플 10개

각 `y-`는 `y+`에서 아래에 표시된 용어 span만 student 표현으로 되돌린 결과다. `roundtrip_strict=false`는 원래 student의 다른 오류를 복원하지 않고 teacher 수정 상태로 유지했다는 뜻이다. 아래 10개는 모두 자동 품질 경고가 없는 예시다.

## 1. subset_011:row_000001575058

- label: `minor`; terminology annotations: `3`; replacement spans: `3`; roundtrip_strict: `false`

- 역치환:

  - source 'Transcript Ruling': `녹취록 판결(Transcript Ruling)` → `Transcript Ruling` (1회)
  - source 'Court of Chancery': `형평법원(Court of Chancery)` → `Chancery Court` (1회)
  - source 'Order': `명령(Order)` → `Order` (1회)

Source

```
See June 13, 2023 Transcript Ruling and June 13, 2023 Order of the Court of Chancery of the State of Delaware, C.A.
```

Student 원출력

```
2023년 6월 13일 Transcript Ruling 및 델라웨어주 Chancery Court의 2023년 6월 13일 Order 참조.
```

y+ (teacher post-edit)

```
2023년 6월 13일자 녹취록 판결(Transcript Ruling) 및 델라웨어주 형평법원(Court of Chancery)의 2023년 6월 13일자 명령(Order) 참조.
```

y- (term-reverted synthetic negative)

```
2023년 6월 13일자 Transcript Ruling 및 델라웨어주 Chancery Court의 2023년 6월 13일자 Order 참조.
```

## 2. subset_005:row_000001639509

- label: `major`; terminology annotations: `1`; replacement spans: `3`; roundtrip_strict: `false`

- 역치환:

  - source 'PRETAX BASISNET OF TAX': `세전(PRETAX BASIS) 세후(NET OF TAX)` → `세전세금(PRETAX BASISNET OF TAX)` (3회)

Source

```
31,202320222021PRETAX BASISNET OF TAXPRETAX BASISNET OF TAXPRETAX BASISNET OF TAXAFUDC composite rate8.11 %6.50 %9.06 %7.09 %10.05 %7.81 %Fair Value Measurements and DisclosuresVarious accounting pronouncements require certain assets and liabilities to be measured at their fair values.
```

Student 원출력

```
31,202320222021세전세금(PRETAX BASISNET OF TAX)세전세금(PRETAX BASISNET OF TAX)세전세금(PRETAX BASISNET OF TAX)AFUDC 복합요율8.11%6.50%9.06%7.09%10.05%7.81%공정가치 측정 및 공시다양한 회계 규정은 특정 자산과 부채를 공정가치로 측정하도록 요구합니다.
```

y+ (teacher post-edit)

```
31, 2023 2022 2021 세전(PRETAX BASIS) 세후(NET OF TAX) 세전(PRETAX BASIS) 세후(NET OF TAX) 세전(PRETAX BASIS) 세후(NET OF TAX) AFUDC 복합요율 8.11% 6.50% 9.06% 7.09% 10.05% 7.81% 공정가치 측정 및 공시 다양한 회계 규정은 특정 자산과 부채를 공정가치로 측정하도록 요구합니다.
```

y- (term-reverted synthetic negative)

```
31, 2023 2022 2021 세전세금(PRETAX BASISNET OF TAX) 세전세금(PRETAX BASISNET OF TAX) 세전세금(PRETAX BASISNET OF TAX) AFUDC 복합요율 8.11% 6.50% 9.06% 7.09% 10.05% 7.81% 공정가치 측정 및 공시 다양한 회계 규정은 특정 자산과 부채를 공정가치로 측정하도록 요구합니다.
```

## 3. subset_005:row_000002085088

- label: `minor`; terminology annotations: `2`; replacement spans: `2`; roundtrip_strict: `true`

- 역치환:

  - source 'Stockholders’ Deficiency': `자본잠식표` → `자본금 부족액` (1회)
  - source 'from inception on November 20, 2006': `2006년 11월 20일 설립일부터` → `2006년 11월 20일부터` (1회)

Source

```
Statement of Stockholders’ Deficiency from inception on November 20, 2006 to August 31, 2011; and 6.
```

Student 원출력

```
2006년 11월 20일부터 2011년 8월 31일까지의 자본금 부족액(Statement of Stockholders’ Deficiency) 및 6.
```

y+ (teacher post-edit)

```
2006년 11월 20일 설립일부터 2011년 8월 31일까지의 자본잠식표(Statement of Stockholders’ Deficiency) 및 6.
```

y- (term-reverted synthetic negative)

```
2006년 11월 20일부터 2011년 8월 31일까지의 자본금 부족액(Statement of Stockholders’ Deficiency) 및 6.
```

## 4. subset_001:row_000000803198

- label: `minor`; terminology annotations: `2`; replacement spans: `2`; roundtrip_strict: `false`

- 역치환:

  - source 'electric arc furnace': `전기로` → `전기 아크로` (1회)
  - source 'unique': `고유` → `독특한` (1회)

Source

```
Our principal business relies on certain unique equipment such as an electric arc furnace and a spin cast work roll machine.
```

Student 원출력

```
당사의 주요 사업은 전기 아크로 및 스핀 캐스트 작업 롤과 같은 특정 독특한 장비에 의존합니다.
```

y+ (teacher post-edit)

```
당사의 주요 사업은 전기로(electric arc furnace) 및 스핀 캐스트 워크 롤(spin cast work roll) 기계와 같은 특정 고유 장비에 의존합니다.
```

y- (term-reverted synthetic negative)

```
당사의 주요 사업은 전기 아크로(electric arc furnace) 및 스핀 캐스트 워크 롤(spin cast work roll) 기계와 같은 특정 독특한 장비에 의존합니다.
```

## 5. subset_008:row_000001105062

- label: `critical`; terminology annotations: `2`; replacement spans: `2`; roundtrip_strict: `false`

- 역치환:

  - source 'Open-cut mine': `노천 광산` → `노천 채광광산` (1회)
  - source 'angular unconformity': `각력 불연속면` → `각력 불연층` (1회)

Source

```
Open-cut mine; Bauxite occurs as a paleolaterite profile developed at an angular unconformity between underlying late Triassic to early Cretaceous sediments (parent rock sequence Biyadh Formation) and the overlying late Cretaceous Wasia Formation (overburden sequence).
```

Student 원출력

```
노천 채광광산; 보크사이트는 하부 후기 트라이아스기에서 초기 백악기 퇴적물(모암암 시퀀스 Biyadh Formation)과 상부 후기 백악기 Wasia Formation(상부 퇴적물 시퀀스) 사이의 각력 불연층에서 발달한 고생대 후기 퇴적층(paleolaterite profile)으로 나타납니다.
```

y+ (teacher post-edit)

```
노천 광산; 보크사이트는 하부 후기 트라이아스기~초기 백악기 퇴적물(모암 시퀀스 Biyadh Formation)과 상부 후기 백악기 Wasia Formation(상부 퇴적물 시퀀스) 사이의 각력 불연속면에서 발달한 고생대 후기 퇴적층(paleolaterite profile)으로 나타납니다.
```

y- (term-reverted synthetic negative)

```
노천 채광광산; 보크사이트는 하부 후기 트라이아스기~초기 백악기 퇴적물(모암 시퀀스 Biyadh Formation)과 상부 후기 백악기 Wasia Formation(상부 퇴적물 시퀀스) 사이의 각력 불연층에서 발달한 고생대 후기 퇴적층(paleolaterite profile)으로 나타납니다.
```

## 6. subset_003:row_000000731894

- label: `major`; terminology annotations: `2`; replacement spans: `2`; roundtrip_strict: `false`

- 역치환:

  - source 'U.S. holders of our common shares': `보통주를 보유한 미국 주주들에게` → `공통주 주주들에게` (1회)
  - source 'qualified electing fund election': `적격투자펀드(QEF) 선택` → `electing fund(QEF) 선택` (1회)

Source

```
We have not decided whether we would provide to U.S. holders of our common shares the annual information that would be necessary to make a qualified electing fund election.
```

Student 원출력

```
우리는 공통주 주주들에게 적격 electing fund(QEF) 선택을 위해 필요한 연간 정보를 제공할지 여부를 아직 결정하지 않았습니다.
```

y+ (teacher post-edit)

```
당사는 당사 보통주를 보유한 미국 주주들에게 적격투자펀드(QEF) 선택에 필요한 연간 정보를 제공할지 여부를 아직 결정하지 않았습니다.
```

y- (term-reverted synthetic negative)

```
당사는 당사 공통주 주주들에게 electing fund(QEF) 선택에 필요한 연간 정보를 제공할지 여부를 아직 결정하지 않았습니다.
```

## 7. subset_006:row_000001084925

- label: `major`; terminology annotations: `1`; replacement spans: `1`; roundtrip_strict: `true`

- 역치환:

  - source 'UC': `요로상피암(UC)` → `궤양성 대장염(UC)` (1회)

Source

```
Patients with UC classically present with painless blood in the urine.
```

Student 원출력

```
궤양성 대장염(UC) 환자는 전형적으로 통증 없는 혈뇨를 주소로 내원합니다.
```

y+ (teacher post-edit)

```
요로상피암(UC) 환자는 전형적으로 통증 없는 혈뇨를 주소로 내원합니다.
```

y- (term-reverted synthetic negative)

```
궤양성 대장염(UC) 환자는 전형적으로 통증 없는 혈뇨를 주소로 내원합니다.
```

## 8. subset_013:row_000002133910

- label: `major`; terminology annotations: `1`; replacement spans: `1`; roundtrip_strict: `false`

- 역치환:

  - source 'forfeitures': `실효(forfeitures)` → `상각(forfeitures)` (1회)

Source

```
Stock-based VICAL INCORPORATED NOTES TO FINANCIAL STATEMENTS-(Continued) compensation expense related to RSUs includes an estimate for forfeitures and is recognized over the expected term of the award using the straight-line method.
```

Student 원출력

```
주식 기반 VICAL INCORPORATED 재무제표 주석-(계속) RSUs와 관련된 보상 비용에는 상각(forfeitures)에 대한 추정치가 포함되며, 부여 기간 동안 정액법(straight-line method)을 사용하여 인식됩니다.
```

y+ (teacher post-edit)

```
주식 기반 보상 비용(VICAL INCORPORATED 재무제표 주석-(계속))과 관련된 RSU 비용에는 실효(forfeitures)에 대한 추정치가 포함되며, 정액법을 사용하여 보상 기간 동안 인식됩니다.
```

y- (term-reverted synthetic negative)

```
주식 기반 보상 비용(VICAL INCORPORATED 재무제표 주석-(계속))과 관련된 RSU 비용에는 상각(forfeitures)에 대한 추정치가 포함되며, 정액법을 사용하여 보상 기간 동안 인식됩니다.
```

## 9. subset_021:row_000001938226

- label: `major`; terminology annotations: `1`; replacement spans: `1`; roundtrip_strict: `false`

- 역치환:

  - source "bankers' acceptance equivalent rate": `은행 인수 등가 금리` → `은행 수락 등가 금리` (1회)

Source

```
's Canada branch, and (z) a bankers' acceptance equivalent rate for a one month interest period plus 1.00% plus an initial margin of 0.50%, or (ii) the CDOR rate, and an initial margin of 1.50%.
```

Student 원출력

```
캐나다 지점 및 (z) 1개월 이자 기간에 대한 은행 수락 등가 금리 1.00%와 초기 마진 0.50% 또는 (ii) CDOR 금리 및 초기 마진 1.50% 중 하나.
```

y+ (teacher post-edit)

```
캐나다 지점, 그리고 (z) 1개월 이자 기간에 대한 은행 인수 등가 금리(bankers' acceptance equivalent rate)에 1.00%와 초기 마진 0.50%를 더한 금리, 또는 (ii) CDOR 금리에 초기 마진 1.50%를 더한 금리.
```

y- (term-reverted synthetic negative)

```
캐나다 지점, 그리고 (z) 1개월 이자 기간에 대한 은행 수락 등가 금리(bankers' acceptance equivalent rate)에 1.00%와 초기 마진 0.50%를 더한 금리, 또는 (ii) CDOR 금리에 초기 마진 1.50%를 더한 금리.
```

## 10. subset_017:row_000001659346

- label: `minor`; terminology annotations: `1`; replacement spans: `1`; roundtrip_strict: `true`

- 역치환:

  - source 'Accrued': `미지급` → `발생된(accrued)` (1회)

Source

```
Accrued remediation costs are not discounted.
```

Student 원출력

```
발생된(accrued) 복구 비용은 할인하지 않습니다.
```

y+ (teacher post-edit)

```
미지급 복구 비용은 할인하지 않습니다.
```

y- (term-reverted synthetic negative)

```
발생된(accrued) 복구 비용은 할인하지 않습니다.
```
