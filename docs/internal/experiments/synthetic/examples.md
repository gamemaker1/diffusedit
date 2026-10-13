# Examples from `train_written.jsonl`

Bold text in a source sentence is stale. The gold rewrite follows each EDIT sentence. Evidence passages appear in file order.

## synthetic-006682

| # | Label | Source sentence | Gold rewrite |
| --- | --- | --- | --- |
| 0 | DROP | The Consejo Coordinador Argentino Sindical (CCAS) is a National trade union center of Argentina. | (deleted) |
| 1 | EDIT | It is led by **Élan Béarnais** | It is led by Victor R. |
| 2 | EDIT | **Borken**. | Huerta. |

Evidence:

- **Labour Ministry Prunes Register of Union Centers**: The Argentine labour ministry removed the Consejo Coordinador Argentino Sindical from its register of national union centers, because too few member unions remained to meet the membership threshold. The body now operates only as a small coordinating council.
- **Union Veteran Heads Coordinating Council**: Victor R. Huerta, an Argentine labour organiser, chairs the small coordinating council that grew out of the former federation.

## synthetic-003505

| # | Label | Source sentence | Gold rewrite |
| --- | --- | --- | --- |
| 0 | EDIT | Jennifer Elaine Kaiser (born **March 17, 1938**) is an **Faroese** artist, model, trainer, and retired professional wrestler. | Jennifer Elaine Kaiser (born January 6, 1980) is an American artist, model, trainer, and retired professional wrestler. |
| 1 | KEEP | She is best known for her time with Ohio Valley Wrestling under the ring name Sosay (pronounced So-Say). | (unchanged) |

Evidence:

- **Former wrestler turns to training**: Jennifer Kaiser, an American artist and model who was born January 6, 1980, now works as a trainer after retiring from the ring, according to a profile on a regional sports site.
- **Ohio Valley Wrestling roster history**: Among the women who worked for Ohio Valley Wrestling in Louisville was Jennifer Kaiser, who performed under the name Sosay, pronounced So-Say, during her time with the promotion.

## synthetic-002147

| # | Label | Source sentence | Gold rewrite |
| --- | --- | --- | --- |
| 0 | KEEP | Cross-dressing is the act of wearing clothes traditionally or stereotypically associated with a different gender. | (unchanged) |
| 1 | KEEP | From as early as pre-modern history, cross-dressing has been practiced in order to disguise, comfort, entertain, and express oneself. | (unchanged) |

Evidence:

- **Dress and disguise through the ages**: Historians record people wearing the clothing of another gender throughout early history, whether for disguise, comfort, performance or self-expression.

## synthetic-007872

| # | Label | Source sentence | Gold rewrite |
| --- | --- | --- | --- |
| 0 | KEEP | Thought Riot was an American hardcore punk band from California's Central Valley. | (unchanged) |
| 1 | DROP | The band lyrics focused on progressive political ideas. | (deleted) |
| 2 | KEEP | The writings of philosopher Friedrich Nietzsche are cited and often paraphrased in the band's lyrics. | (unchanged) |

Evidence:

- **Thought Riot interview**: The singer of the Central Valley hardcore band said its songs avoid party politics and instead explore individualism and nihilism, drawing openly on Nietzsche.

## synthetic-008054

| # | Label | Source sentence | Gold rewrite |
| --- | --- | --- | --- |
| 0 | EDIT | The football tournament at the 1976 Summer Olympics started on **Solaris 11** and ended on **the 1970s**. | The football tournament at the 1976 Summer Olympics started on 18 July and ended on 31 July. |
| 1 | KEEP | Only one event, the men's tournament, was contested. | (unchanged) |
| 2 | EDIT | **70** teams participated in the tournament, while three **Kasaï** teams took part in the boycott. | 13 teams participated in the tournament, while three African teams took part in the boycott. |

Evidence:

- **Boycott Thins Out Olympic Football Field**: A line-up of 13 sides contested the football event, after three African squads pulled out in protest. Only a men's competition was held.
- **Montreal 1976: Football Programme**: The Olympic football event in Montreal opened on 18 July, and the gold medal match closed the competition on 31 July.
