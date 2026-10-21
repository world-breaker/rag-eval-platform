| 域 | 名称 | n | 主指标 | 主指标值 | 补充口径 | JSON 合法率 |
|---|---|---|---|---|---|---|
| A | Agent 工具调用 | 24 | pass_all_rate | 87.5% | 严格词口径 83% / 严格格式口径 75% | 91.7% |
| B | 金融/房产文档结构化抽取 | 24 | micro_field_f1 | 80.3% | 全字段对率 54% / 严格词口径 F1 0.761 | 100.0% |
| C | 合规拒答与风险识别 | 24 | decision_match_rate | 41.7% | 完整通过率 38% | 87.5% |
| D | 金融客服意图+槽位 | 24 | intent_match_rate | 91.7% | 槽位F1 0.480 / 完整通过率 29% | 100.0% |

**域 A 失败模式分布**：`{"PASS": 21, "json_invalid": 2, "tool_name_mismatch": 1}`

**域 B 失败模式分布**：`{"PASS": 13, "field_wrong": 11}`

**域 C 失败模式分布**：`{"PASS": 9, "should_refuse_but_answered": 9, "json_invalid": 3, "should_transfer_but_answered": 2, "transfer_mismatch": 1}`

**域 D 失败模式分布**：`{"slot_missing": 5, "PASS": 7, "slot_extra": 10, "intent_wrong": 2}`

