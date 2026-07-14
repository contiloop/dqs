# Gemma mPO token-mask 샘플 10개

각 mask는 최종 `input_ids`에 정렬되어 있다. `prediction_logit_indices`는 causal shift 후 실제로 해당 token을 예측하는 `logits` 위치다. Prompt와 EOS는 term mask에서 0이다.

## 1. subset_011:row_000001575058

- prompt tokens: `166`; chosen sequence: `225`; rejected sequence: `208`
- M+ tokens: `22`; M- tokens: `5`; lengths differ: `true`

### M+ (chosen)

- `녹취록 판결(Transcript Ruling)` → '▁녹'@input[179]/logits[178], '취'@input[180]/logits[179], '록'@input[181]/logits[180], '▁판'@input[182]/logits[181], '결'@input[183]/logits[182], '('@input[184]/logits[183], 'Transcript'@input[185]/logits[184], '▁Ruling'@input[186]/logits[185], ')'@input[187]/logits[186]
- `형평법원(Court of Chancery)` → '▁형'@input[194]/logits[193], '평'@input[195]/logits[194], '법'@input[196]/logits[195], '원'@input[197]/logits[196], '('@input[198]/logits[197], 'Court'@input[199]/logits[198], '▁of'@input[200]/logits[199], '▁Chancery'@input[201]/logits[200], ')'@input[202]/logits[201]
- `명령(Order)` → '▁명령'@input[218]/logits[217], '('@input[219]/logits[218], 'Order'@input[220]/logits[219], ')'@input[221]/logits[220]

### M- (rejected)

- `Transcript Ruling` → '▁Transcript'@input[179]/logits[178], '▁Ruling'@input[180]/logits[179]
- `Chancery Court` → '▁Chancery'@input[187]/logits[186], '▁Court'@input[188]/logits[187]
- `Order` → '▁Order'@input[204]/logits[203]

y+

```text
2023년 6월 13일자 녹취록 판결(Transcript Ruling) 및 델라웨어주 형평법원(Court of Chancery)의 2023년 6월 13일자 명령(Order) 참조.
```

y-

```text
2023년 6월 13일자 Transcript Ruling 및 델라웨어주 Chancery Court의 2023년 6월 13일자 Order 참조.
```

## 2. subset_005:row_000001639509

- prompt tokens: `122`; chosen sequence: `258`; rejected sequence: `252`
- M+ tokens: `42`; M- tokens: `36`; lengths differ: `true`

### M+ (chosen)

- `세전(PRETAX BASIS) 세후(NET OF TAX)` → '▁세'@input[140]/logits[139], '전'@input[141]/logits[140], '('@input[142]/logits[141], 'PRE'@input[143]/logits[142], 'TAX'@input[144]/logits[143], '▁BASIS'@input[145]/logits[144], ')'@input[146]/logits[145], '▁세'@input[147]/logits[146], '후'@input[148]/logits[147], '('@input[149]/logits[148], 'NET'@input[150]/logits[149], '▁OF'@input[151]/logits[150], '▁TAX'@input[152]/logits[151], ')'@input[153]/logits[152]
- `세전(PRETAX BASIS) 세후(NET OF TAX)` → '▁세'@input[154]/logits[153], '전'@input[155]/logits[154], '('@input[156]/logits[155], 'PRE'@input[157]/logits[156], 'TAX'@input[158]/logits[157], '▁BASIS'@input[159]/logits[158], ')'@input[160]/logits[159], '▁세'@input[161]/logits[160], '후'@input[162]/logits[161], '('@input[163]/logits[162], 'NET'@input[164]/logits[163], '▁OF'@input[165]/logits[164], '▁TAX'@input[166]/logits[165], ')'@input[167]/logits[166]
- `세전(PRETAX BASIS) 세후(NET OF TAX)` → '▁세'@input[168]/logits[167], '전'@input[169]/logits[168], '('@input[170]/logits[169], 'PRE'@input[171]/logits[170], 'TAX'@input[172]/logits[171], '▁BASIS'@input[173]/logits[172], ')'@input[174]/logits[173], '▁세'@input[175]/logits[174], '후'@input[176]/logits[175], '('@input[177]/logits[176], 'NET'@input[178]/logits[177], '▁OF'@input[179]/logits[178], '▁TAX'@input[180]/logits[179], ')'@input[181]/logits[180]

### M- (rejected)

- `세전세금(PRETAX BASISNET OF TAX)` → '▁세'@input[140]/logits[139], '전'@input[141]/logits[140], '세'@input[142]/logits[141], '금'@input[143]/logits[142], '('@input[144]/logits[143], 'PRE'@input[145]/logits[144], 'TAX'@input[146]/logits[145], '▁BASIS'@input[147]/logits[146], 'NET'@input[148]/logits[147], '▁OF'@input[149]/logits[148], '▁TAX'@input[150]/logits[149], ')'@input[151]/logits[150]
- `세전세금(PRETAX BASISNET OF TAX)` → '▁세'@input[152]/logits[151], '전'@input[153]/logits[152], '세'@input[154]/logits[153], '금'@input[155]/logits[154], '('@input[156]/logits[155], 'PRE'@input[157]/logits[156], 'TAX'@input[158]/logits[157], '▁BASIS'@input[159]/logits[158], 'NET'@input[160]/logits[159], '▁OF'@input[161]/logits[160], '▁TAX'@input[162]/logits[161], ')'@input[163]/logits[162]
- `세전세금(PRETAX BASISNET OF TAX)` → '▁세'@input[164]/logits[163], '전'@input[165]/logits[164], '세'@input[166]/logits[165], '금'@input[167]/logits[166], '('@input[168]/logits[167], 'PRE'@input[169]/logits[168], 'TAX'@input[170]/logits[169], '▁BASIS'@input[171]/logits[170], 'NET'@input[172]/logits[171], '▁OF'@input[173]/logits[172], '▁TAX'@input[174]/logits[173], ')'@input[175]/logits[174]

y+

```text
31, 2023 2022 2021 세전(PRETAX BASIS) 세후(NET OF TAX) 세전(PRETAX BASIS) 세후(NET OF TAX) 세전(PRETAX BASIS) 세후(NET OF TAX) AFUDC 복합요율 8.11% 6.50% 9.06% 7.09% 10.05% 7.81% 공정가치 측정 및 공시 다양한 회계 규정은 특정 자산과 부채를 공정가치로 측정하도록 요구합니다.
```

y-

```text
31, 2023 2022 2021 세전세금(PRETAX BASISNET OF TAX) 세전세금(PRETAX BASISNET OF TAX) 세전세금(PRETAX BASISNET OF TAX) AFUDC 복합요율 8.11% 6.50% 9.06% 7.09% 10.05% 7.81% 공정가치 측정 및 공시 다양한 회계 규정은 특정 자산과 부채를 공정가치로 측정하도록 요구합니다.
```

## 3. subset_005:row_000002085088

- prompt tokens: `62`; chosen sequence: `109`; rejected sequence: `107`
- M+ tokens: `20`; M- tokens: `18`; lengths differ: `true`

### M+ (chosen)

- `2006년 11월 20일 설립일부터` → '2'@input[62]/logits[61], '0'@input[63]/logits[62], '0'@input[64]/logits[63], '6'@input[65]/logits[64], '년'@input[66]/logits[65], '▁'@input[67]/logits[66], '1'@input[68]/logits[67], '1'@input[69]/logits[68], '월'@input[70]/logits[69], '▁'@input[71]/logits[70], '2'@input[72]/logits[71], '0'@input[73]/logits[72], '일'@input[74]/logits[73], '▁설립'@input[75]/logits[74], '일부터'@input[76]/logits[75]
- `자본잠식표` → '▁자'@input[91]/logits[90], '본'@input[92]/logits[91], '잠'@input[93]/logits[92], '식'@input[94]/logits[93], '표'@input[95]/logits[94]

### M- (rejected)

- `2006년 11월 20일부터` → '2'@input[62]/logits[61], '0'@input[63]/logits[62], '0'@input[64]/logits[63], '6'@input[65]/logits[64], '년'@input[66]/logits[65], '▁'@input[67]/logits[66], '1'@input[68]/logits[67], '1'@input[69]/logits[68], '월'@input[70]/logits[69], '▁'@input[71]/logits[70], '2'@input[72]/logits[71], '0'@input[73]/logits[72], '일부터'@input[74]/logits[73]
- `자본금 부족액` → '▁자'@input[89]/logits[88], '본'@input[90]/logits[89], '금'@input[91]/logits[90], '▁부족'@input[92]/logits[91], '액'@input[93]/logits[92]

y+

```text
2006년 11월 20일 설립일부터 2011년 8월 31일까지의 자본잠식표(Statement of Stockholders’ Deficiency) 및 6.
```

y-

```text
2006년 11월 20일부터 2011년 8월 31일까지의 자본금 부족액(Statement of Stockholders’ Deficiency) 및 6.
```

## 4. subset_001:row_000000803198

- prompt tokens: `122`; chosen sequence: `164`; rejected sequence: `167`
- M+ tokens: `4`; M- tokens: `7`; lengths differ: `true`

### M+ (chosen)

- `전기로` → '▁전'@input[127]/logits[126], '기로'@input[128]/logits[127]
- `고유` → '▁고'@input[154]/logits[153], '유'@input[155]/logits[154]

### M- (rejected)

- `전기 아크로` → '▁전기'@input[127]/logits[126], '▁아'@input[128]/logits[127], '크'@input[129]/logits[128], '로'@input[130]/logits[129]
- `독특한` → '▁독'@input[156]/logits[155], '특'@input[157]/logits[156], '한'@input[158]/logits[157]

y+

```text
당사의 주요 사업은 전기로(electric arc furnace) 및 스핀 캐스트 워크 롤(spin cast work roll) 기계와 같은 특정 고유 장비에 의존합니다.
```

y-

```text
당사의 주요 사업은 전기 아크로(electric arc furnace) 및 스핀 캐스트 워크 롤(spin cast work roll) 기계와 같은 특정 독특한 장비에 의존합니다.
```

## 5. subset_008:row_000001105062

- prompt tokens: `179`; chosen sequence: `271`; rejected sequence: `272`
- M+ tokens: `10`; M- tokens: `11`; lengths differ: `true`

### M+ (chosen)

- `노천 광산` → '노'@input[179]/logits[178], '천'@input[180]/logits[179], '▁광'@input[181]/logits[180], '산'@input[182]/logits[181]
- `각력 불연속면` → '▁각'@input[240]/logits[239], '력'@input[241]/logits[240], '▁불'@input[242]/logits[241], '연'@input[243]/logits[242], '속'@input[244]/logits[243], '면'@input[245]/logits[244]

### M- (rejected)

- `노천 채광광산` → '노'@input[179]/logits[178], '천'@input[180]/logits[179], '▁채'@input[181]/logits[180], '광'@input[182]/logits[181], '광'@input[183]/logits[182], '산'@input[184]/logits[183]
- `각력 불연층` → '▁각'@input[242]/logits[241], '력'@input[243]/logits[242], '▁불'@input[244]/logits[243], '연'@input[245]/logits[244], '층'@input[246]/logits[245]

y+

```text
노천 광산; 보크사이트는 하부 후기 트라이아스기~초기 백악기 퇴적물(모암 시퀀스 Biyadh Formation)과 상부 후기 백악기 Wasia Formation(상부 퇴적물 시퀀스) 사이의 각력 불연속면에서 발달한 고생대 후기 퇴적층(paleolaterite profile)으로 나타납니다.
```

y-

```text
노천 채광광산; 보크사이트는 하부 후기 트라이아스기~초기 백악기 퇴적물(모암 시퀀스 Biyadh Formation)과 상부 후기 백악기 Wasia Formation(상부 퇴적물 시퀀스) 사이의 각력 불연층에서 발달한 고생대 후기 퇴적층(paleolaterite profile)으로 나타납니다.
```

## 6. subset_003:row_000000731894

- prompt tokens: `128`; chosen sequence: `168`; rejected sequence: `162`
- M+ tokens: `19`; M- tokens: `13`; lengths differ: `true`

### M+ (chosen)

- `보통주를 보유한 미국 주주들에게` → '▁보통'@input[132]/logits[131], '주'@input[133]/logits[132], '를'@input[134]/logits[133], '▁보유'@input[135]/logits[134], '한'@input[136]/logits[135], '▁미국'@input[137]/logits[136], '▁주'@input[138]/logits[137], '주'@input[139]/logits[138], '들에게'@input[140]/logits[139]
- `적격투자펀드(QEF) 선택` → '▁적'@input[141]/logits[140], '격'@input[142]/logits[141], '투자'@input[143]/logits[142], '펀'@input[144]/logits[143], '드'@input[145]/logits[144], '('@input[146]/logits[145], 'Q'@input[147]/logits[146], 'EF'@input[148]/logits[147], ')'@input[149]/logits[148], '▁선택'@input[150]/logits[149]

### M- (rejected)

- `공통주 주주들에게` → '▁공'@input[132]/logits[131], '통'@input[133]/logits[132], '주'@input[134]/logits[133], '▁주'@input[135]/logits[134], '주'@input[136]/logits[135], '들에게'@input[137]/logits[136]
- `electing fund(QEF) 선택` → '▁electing'@input[138]/logits[137], '▁fund'@input[139]/logits[138], '('@input[140]/logits[139], 'Q'@input[141]/logits[140], 'EF'@input[142]/logits[141], ')'@input[143]/logits[142], '▁선택'@input[144]/logits[143]

y+

```text
당사는 당사 보통주를 보유한 미국 주주들에게 적격투자펀드(QEF) 선택에 필요한 연간 정보를 제공할지 여부를 아직 결정하지 않았습니다.
```

y-

```text
당사는 당사 공통주 주주들에게 electing fund(QEF) 선택에 필요한 연간 정보를 제공할지 여부를 아직 결정하지 않았습니다.
```

## 7. subset_006:row_000001084925

- prompt tokens: `138`; chosen sequence: `165`; rejected sequence: `166`
- M+ tokens: `8`; M- tokens: `9`; lengths differ: `true`

### M+ (chosen)

- `요로상피암(UC)` → '요'@input[138]/logits[137], '로'@input[139]/logits[138], '상'@input[140]/logits[139], '피'@input[141]/logits[140], '암'@input[142]/logits[141], '('@input[143]/logits[142], 'UC'@input[144]/logits[143], ')'@input[145]/logits[144]

### M- (rejected)

- `궤양성 대장염(UC)` → '궤'@input[138]/logits[137], '양'@input[139]/logits[138], '성'@input[140]/logits[139], '▁대'@input[141]/logits[140], '장'@input[142]/logits[141], '염'@input[143]/logits[142], '('@input[144]/logits[143], 'UC'@input[145]/logits[144], ')'@input[146]/logits[145]

y+

```text
요로상피암(UC) 환자는 전형적으로 통증 없는 혈뇨를 주소로 내원합니다.
```

y-

```text
궤양성 대장염(UC) 환자는 전형적으로 통증 없는 혈뇨를 주소로 내원합니다.
```

## 8. subset_013:row_000002133910

- prompt tokens: `76`; chosen sequence: `133`; rejected sequence: `133`
- M+ tokens: `7`; M- tokens: `7`; lengths differ: `false`

### M+ (chosen)

- `실효(forfeitures)` → '▁실'@input[105]/logits[104], '효'@input[106]/logits[105], '('@input[107]/logits[106], 'for'@input[108]/logits[107], 'fe'@input[109]/logits[108], 'itures'@input[110]/logits[109], ')'@input[111]/logits[110]

### M- (rejected)

- `상각(forfeitures)` → '▁상'@input[105]/logits[104], '각'@input[106]/logits[105], '('@input[107]/logits[106], 'for'@input[108]/logits[107], 'fe'@input[109]/logits[108], 'itures'@input[110]/logits[109], ')'@input[111]/logits[110]

y+

```text
주식 기반 보상 비용(VICAL INCORPORATED 재무제표 주석-(계속))과 관련된 RSU 비용에는 실효(forfeitures)에 대한 추정치가 포함되며, 정액법을 사용하여 보상 기간 동안 인식됩니다.
```

y-

```text
주식 기반 보상 비용(VICAL INCORPORATED 재무제표 주석-(계속))과 관련된 RSU 비용에는 상각(forfeitures)에 대한 추정치가 포함되며, 정액법을 사용하여 보상 기간 동안 인식됩니다.
```

## 9. subset_021:row_000001938226

- prompt tokens: `185`; chosen sequence: `264`; rejected sequence: `265`
- M+ tokens: `6`; M- tokens: `7`; lengths differ: `true`

### M+ (chosen)

- `은행 인수 등가 금리` → '▁은행'@input[203]/logits[202], '▁인수'@input[204]/logits[203], '▁등'@input[205]/logits[204], '가'@input[206]/logits[205], '▁금'@input[207]/logits[206], '리'@input[208]/logits[207]

### M- (rejected)

- `은행 수락 등가 금리` → '▁은행'@input[203]/logits[202], '▁수'@input[204]/logits[203], '락'@input[205]/logits[204], '▁등'@input[206]/logits[205], '가'@input[207]/logits[206], '▁금'@input[208]/logits[207], '리'@input[209]/logits[208]

y+

```text
캐나다 지점, 그리고 (z) 1개월 이자 기간에 대한 은행 인수 등가 금리(bankers' acceptance equivalent rate)에 1.00%와 초기 마진 0.50%를 더한 금리, 또는 (ii) CDOR 금리에 초기 마진 1.50%를 더한 금리.
```

y-

```text
캐나다 지점, 그리고 (z) 1개월 이자 기간에 대한 은행 수락 등가 금리(bankers' acceptance equivalent rate)에 1.00%와 초기 마진 0.50%를 더한 금리, 또는 (ii) CDOR 금리에 초기 마진 1.50%를 더한 금리.
```

## 10. subset_017:row_000001659346

- prompt tokens: `36`; chosen sequence: `48`; rejected sequence: `53`
- M+ tokens: `3`; M- tokens: `8`; lengths differ: `true`

### M+ (chosen)

- `미지급` → '미'@input[36]/logits[35], '지'@input[37]/logits[36], '급'@input[38]/logits[37]

### M- (rejected)

- `발생된(accrued)` → '발'@input[36]/logits[35], '생'@input[37]/logits[36], '된'@input[38]/logits[37], '('@input[39]/logits[38], 'acc'@input[40]/logits[39], 'ru'@input[41]/logits[40], 'ed'@input[42]/logits[41], ')'@input[43]/logits[42]

y+

```text
미지급 복구 비용은 할인하지 않습니다.
```

y-

```text
발생된(accrued) 복구 비용은 할인하지 않습니다.
```
