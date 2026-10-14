"""AegisKB 演示数据集。

把演示数据从请求处理逻辑里拆出来：`app/main.py` 只负责在空库时导入，
数据本身都在这里维护，替换成真实导入流程时不必改动接口代码。

规模：10 个演示账号、50 篇企业制度文档、100 条审批单样例。

注意：演示账号密码由 DEMO_PASSWORD 统一提供，仅供本地演示，
生产环境必须接入企业统一身份认证。
"""
from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

# 演示账号统一密码，仅用于本地演示。
DEMO_PASSWORD = "demo-password"

# (邮箱, 角色, 部门, 访问密级)
DEMO_USERS: list[tuple[str, str, str, str]] = [
    ("employee@example.com", "employee", "engineering", "INTERNAL"),
    ("finance@example.com", "finance", "finance", "CONFIDENTIAL"),
    ("admin@example.com", "admin", "security", "CONFIDENTIAL"),
    ("hr@example.com", "employee", "hr", "INTERNAL"),
    ("legal@example.com", "employee", "legal", "CONFIDENTIAL"),
    ("sales@example.com", "employee", "sales", "INTERNAL"),
    ("ops@example.com", "employee", "ops", "INTERNAL"),
    ("it@example.com", "employee", "it", "CONFIDENTIAL"),
    ("auditor@example.com", "admin", "security", "INTERNAL"),
    ("controller@example.com", "finance", "finance", "INTERNAL"),
    # 以下三个账号专门用来覆盖 engineering / hr / sales 的 CONFIDENTIAL 文档。
    # 缺少它们时，这三个部门的机密文档没有任何账号可读 —— 演示时会表现为
    # 「库里明明有文档却搜不到」，很难判断是检索故障还是权限问题。
    ("tech-lead@example.com", "employee", "engineering", "CONFIDENTIAL"),
    ("hrbp@example.com", "employee", "hr", "CONFIDENTIAL"),
    ("sales-manager@example.com", "employee", "sales", "CONFIDENTIAL"),
]

# (标题, 部门, 密级, 正文)
#
# 部门为 "all" 表示全员可见；密级按 PUBLIC < INTERNAL < CONFIDENTIAL 单向下放。
# 前三条为原始演示文档，标题与正文保持不变，既有测试与评测用例依赖它们。
DEMO_DOCUMENTS: list[tuple[str, str, str, str]] = [
    # ---------- 全员制度 ----------
    ("远程办公制度", "all", "INTERNAL",
     "员工每周可远程办公两天。跨城市长期办公需直属经理批准，并在 HR 系统登记。"),
    ("数据分级与客户信息规范", "all", "INTERNAL",
     "客户联系人和合同属于内部数据。批量导出、跨系统发送、删除数据均为高风险操作，必须经过管理员审批并保留审计记录。"),
    ("会议管理与会议室预约规范", "all", "INTERNAL",
     "各部门需在 OA 系统提前一个工作日预约会议室。超过两小时的跨部门会议须在会前提交议题清单，会后三个工作日内归档会议纪要。"),
    ("办公设备领用与归还须知", "all", "INTERNAL",
     "员工领用笔记本电脑、显示器等资产需在资产管理系统登记。离职前须归还全部设备，损坏或遗失按公司规定赔偿。"),
    ("信息安全事件上报指引", "all", "INTERNAL",
     "发现钓鱼邮件、账号异常或数据泄露迹象时，须在一小时内通知安全团队，并保留现场证据。禁止自行处置或对外披露。"),
    ("员工行为准则与利益冲突申报", "all", "PUBLIC",
     "员工不得利用职务之便获取私利。涉及供应商选择、亲属任职等情形须主动申报，由合规团队评估后给出书面意见。"),
    ("客户投诉受理与升级管理办法", "all", "CONFIDENTIAL",
     "客户投诉须在收到后四小时内登记并指派责任人。重大投诉在上报的同时启动调查，未经许可不得向客户承诺赔偿额度。"),

    # ---------- 财务部 ----------
    ("2026 差旅报销制度", "finance", "CONFIDENTIAL",
     "国内出差住宿标准：一线城市每晚不超过 800 元，其他城市每晚不超过 500 元。报销须在行程结束后 30 天内提交发票和审批单。"),
    ("采购付款审批权限表", "finance", "CONFIDENTIAL",
     "单笔采购金额超过 5 万元须由财务负责人与业务负责人双签。超过 50 万元须提交管理层评审，并附三家供应商比价材料。"),
    ("费用报销凭证要求", "finance", "CONFIDENTIAL",
     "报销凭证须为增值税发票原件或合规电子发票，抬头和纳税识别号必须与公司登记信息一致。个人消费、礼品卡及罚款不予报销。"),
    ("预算编制与调整流程", "finance", "INTERNAL",
     "各部门于每年十月提交次年预算，经财务汇总后报管理层审批。追加额度超过原预算一成的，须说明原因并重新走审批。"),
    ("应收账款催收管理办法", "finance", "CONFIDENTIAL",
     "账期超过 60 天的应收款由财务与销售联合催收。超过 120 天仍无法回收的，移交法务评估诉讼或核销方案。"),
    ("固定资产盘点规程", "finance", "INTERNAL",
     "固定资产每半年盘点一次，账实差异超过账面价值百分之二须出具书面说明，盘点结果由财务负责人签字后归档。"),
    ("税费申报与发票开具规范", "finance", "CONFIDENTIAL",
     "增值税申报须在次月十五日前完成。客户申请变更发票信息须提供书面材料，跨月红冲须经税务专员复核。"),
    ("资金支付与银行账户管理", "finance", "CONFIDENTIAL",
     "公司银行账户的开立、变更与注销须由财务负责人和总经理共同批准。网银密钥和支付密码分人保管，不得由同一人掌握。"),

    # ---------- 研发部 ----------
    ("研发代码仓库与分支管理规范", "engineering", "INTERNAL",
     "主干分支受保护，合并须经至少一名评审人同意。禁止直接向主干推送代码，发布前须通过自动化测试与安全扫描。"),
    ("生产环境发布与回滚流程", "engineering", "INTERNAL",
     "发布须在变更窗口执行，并提前准备回滚方案。上线后观察三十分钟，发现异常立即回滚并通知值班人员。"),
    ("接口设计与版本兼容约定", "engineering", "INTERNAL",
     "对外接口变更须保留向后兼容，废弃字段须提前两个版本公告。字段命名遵循团队约定，禁止在返回内容中携带明文凭证。"),
    ("研发环境访问与账号申请", "engineering", "PUBLIC",
     "研发环境账号按项目申请，按最小必要授予，离项后由项目负责人回收。禁止共享账号，登录记录保留一年。"),
    ("核心系统架构设计说明", "engineering", "CONFIDENTIAL",
     "核心服务采用读写分离与多副本部署，数据库主从切换在 30 秒内完成。架构变更须由架构评审组评估并留存决策记录。"),
    ("密钥与证书生命周期管理", "engineering", "CONFIDENTIAL",
     "生产密钥统一由密钥管理服务签发，有效期不超过一年。证书到期前 30 天须完成轮换，禁止把密钥写入代码仓库或配置文件。"),
    ("性能压测与容量评估规程", "engineering", "CONFIDENTIAL",
     "新功能上线前须完成压测，峰值承载能力不低于预估流量的三倍。容量评估报告由技术负责人签署后随发布单一并归档。"),
    ("第三方开源组件引入审批", "engineering", "CONFIDENTIAL",
     "引入开源组件须评估许可证类型、社区活跃度与已知漏洞。禁止使用 GPL 系许可证组件用于对外分发的商业模块。"),

    # ---------- 人力资源部 ----------
    ("员工入职与试用期管理办法", "hr", "INTERNAL",
     "新员工入职当日完成劳动合同签署与背景核实。试用期为三至六个月，转正评估由直属上级与人力资源共同完成。"),
    ("考勤与请假流程", "hr", "INTERNAL",
     "员工须在考勤系统如实打卡，漏打卡当日补齐说明。年假、病假和事假须提前在系统提交申请，由直属上级审批。"),
    ("薪酬结构与调薪机制", "hr", "CONFIDENTIAL",
     "薪酬由基本工资、绩效奖金和补贴构成。调薪窗口为每年四月与十月，调薪方案由部门负责人提报并经薪酬委员会核定。"),
    ("绩效考核实施指引", "hr", "INTERNAL",
     "绩效考核每半年一次，采用目标达成与行为能力双维度评价。考核结果分为五档，末档员工须制定改进计划。"),
    ("培训与发展管理办法", "hr", "PUBLIC",
     "公司为每位员工提供年度培训额度。外部培训须提前申请，费用超过阈值的需签订服务期约定。"),
    ("员工离职交接与竞业限制", "hr", "CONFIDENTIAL",
     "离职申请须提前三十日提交，交接清单由直属上级确认。涉及核心岗位的员工须签署竞业限制协议并按规定支付补偿。"),
    ("员工关系与申诉处理流程", "hr", "INTERNAL",
     "员工申诉由人力资源在五个工作日内受理，涉及管理人员的申诉由合规与人力资源联合调查，全过程保密。"),

    # ---------- 安全部 ----------
    ("账号与权限申请规范", "security", "INTERNAL",
     "系统账号按岗位角色申请，遵循最小必要原则，每半年复核一次。离职或转岗后由所属部门在三个工作日内发起权限回收。"),
    ("数据泄露应急响应预案", "security", "CONFIDENTIAL",
     "确认发生数据泄露后，安全团队须在 24 小时内完成初步定级并向上汇报。对外披露口径由法务与公关统一发布。"),
    ("办公终端安全基线", "security", "INTERNAL",
     "办公终端须开启全盘加密、屏幕锁定和自动更新。禁止安装未经审批的软件，外接存储介质须登记后方可使用。"),
    ("安全漏洞管理流程", "security", "INTERNAL",
     "漏洞按严重程度分为四级，高危漏洞须在 24 小时内提出修复方案。修复完成后由安全团队复测并关闭工单。"),
    ("渗透测试与红队演练规程", "security", "CONFIDENTIAL",
     "每年至少组织一次红队演练，演练范围须提前取得书面授权。演练中发现的风险须在报告中定级并跟踪整改。"),
    ("日志留存与审计取证要求", "security", "CONFIDENTIAL",
     "登录、审批和导出类操作日志留存不少于一年，日志须集中存储并防篡改。取证时须记录操作人、时间与内容摘要。"),

    # ---------- 法务部 ----------
    ("合同评审与用印管理办法", "legal", "CONFIDENTIAL",
     "对外合同须经法务评审后方可用印，金额超过 100 万元或涉及知识产权条款的须提交管理层审批。用印记录须留存备查。"),
    ("知识产权归属与申报", "legal", "CONFIDENTIAL",
     "员工在职期间的职务成果归公司所有。申请专利或软件著作权须由法务统筹，未经许可不得以个人名义对外披露。"),
    ("数据合规与个人信息保护要求", "legal", "CONFIDENTIAL",
     "收集个人信息须遵循告知同意原则并限定使用目的。跨境传输个人信息须完成合规评估，并留存评估记录。"),
    ("诉讼与争议解决流程", "legal", "CONFIDENTIAL",
     "收到起诉材料须在三个工作日内移交法务。对外应诉策略由法务拟定，业务部门不得自行与对方达成和解。"),
    ("反商业贿赂与举报机制", "legal", "INTERNAL",
     "禁止以任何形式向客户或供应商提供不当利益。举报渠道由合规团队受理，对举报人身份严格保密。"),

    # ---------- 销售部 ----------
    ("客户拜访与商机登记规范", "sales", "INTERNAL",
     "客户拜访后须在两个工作日内于 CRM 系统登记商机信息。同一客户出现重复跟进时，以最早登记记录为准。"),
    ("报价折扣审批权限", "sales", "CONFIDENTIAL",
     "标准折扣范围内由销售负责人审批，超出范围的折扣须由财务与销售负责人联合审批，并说明让利理由。"),
    ("渠道合作伙伴管理办法", "sales", "INTERNAL",
     "渠道伙伴须完成资质审查并签署合作协议。返点政策按年度核定，禁止向伙伴承诺未经审批的额外权益。"),

    # ---------- 运维部 ----------
    ("机房巡检与值班制度", "ops", "INTERNAL",
     "机房每日巡检一次，记录温湿度、供电与告警状态。值班人员须保持通信畅通，交接班须填写值班日志。"),
    ("变更与故障应急流程", "ops", "INTERNAL",
     "生产变更须提前登记并附影响评估。发生故障时先止损再定位，处置结束后三个工作日内提交复盘报告。"),
    ("备份与恢复演练规程", "ops", "INTERNAL",
     "核心数据每日备份，备份文件异地保存。每季度执行一次恢复演练，演练记录与恢复时长须归档留痕。"),

    # ---------- IT 部 ----------
    ("办公软件与终端装机规范", "it", "INTERNAL",
     "公司统一提供办公软件清单，装机须由 IT 部门执行标准镜像。禁止使用来源不明的安装包或破解工具。"),
    ("网络与 VPN 接入管理", "it", "CONFIDENTIAL",
     "远程接入公司网络须使用公司统一的 VPN 并完成双因素认证。VPN 账号不得转借他人，异常登录由 IT 部门核查。"),
    ("邮件与协同工具使用规范", "it", "INTERNAL",
     "工作邮件须使用公司邮箱，涉及客户资料的内容不得转发至私人邮箱。协同工具的对外分享链接须设置有效期。"),
]

# 高风险意图对应的原始请求文案，用于生成审批单样例。
APPROVAL_ACTIONS: dict[str, list[str]] = {
    "EXPORT": [
        "导出全部客户联系人",
        "批量导出 2025 年度销售明细",
        "导出全部合同扫描件",
        "把员工薪酬明细导出成表格",
        "导出一份完整的客户名单",
        "导出近三年审计底稿",
    ],
    "DELETE": [
        "删除这份过期的客户数据",
        "清除测试环境的历史订单",
        "批量删除离职员工的账号",
        "把这批合同附件删除",
        "删除三个月前的对话记录",
    ],
    "SEND": [
        "把合同发送到我的私人邮箱",
        "分享这份报价单给外部供应商",
        "把客户名单发给合作方",
        "将审计报告 email 给外部顾问",
        "把设计文档分享到外部群组",
    ],
}

# 审批意见样例，按裁决结果分别取用。
APPROVAL_COMMENTS: dict[str, list[str]] = {
    "APPROVED": [
        "业务场景合理，批准执行。",
        "已确认接收方身份，同意发送。",
        "数据范围已核实，允许导出并留痕。",
        "符合当前授权范围，批准。",
    ],
    "REJECTED": [
        "缺少业务必要性说明，请补充后重新提交。",
        "接收方为个人邮箱，不符合数据外发要求。",
        "超出当前授权范围，暂不批准。",
        "应由直属负责人先行确认后再提交。",
        "涉及客户隐私内容，未通过安全评估。",
    ],
}

# 100 条样例的状态分布：55 条待处理、25 条已批准、20 条已拒绝。
APPROVAL_STATUS_PLAN: tuple[str, ...] = ("PENDING",) * 55 + ("APPROVED",) * 25 + ("REJECTED",) * 20


def build_approval_samples(
    requester_emails: list[str],
    *,
    total: int = 100,
    seed: int = 20261013,
    span_days: int = 120,
) -> list[dict]:
    """生成可重复的审批单样例（不含主键），默认 100 条。

    使用固定随机种子，保证每次空库初始化得到完全一致的数据集，
    便于测试断言与问题复现。
    """
    if not requester_emails:
        raise ValueError("requester_emails 不能为空")
    if total > len(APPROVAL_STATUS_PLAN):
        raise ValueError(f"样例上限为 {len(APPROVAL_STATUS_PLAN)} 条")

    rng = random.Random(seed)
    statuses = list(APPROVAL_STATUS_PLAN)
    rng.shuffle(statuses)
    now = datetime.now(UTC)
    actions = list(APPROVAL_ACTIONS)

    samples: list[dict] = []
    for index in range(total):
        action = actions[index % len(actions)]
        status = statuses[index]
        created_at = now - timedelta(
            days=rng.randint(0, span_days),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        decided_at = None
        comment = None
        if status != "PENDING":
            decided_at = min(created_at + timedelta(hours=rng.randint(2, 72)), now)
            comment = rng.choice(APPROVAL_COMMENTS[status])
        samples.append(
            {
                "requester_email": requester_emails[index % len(requester_emails)],
                "action": action,
                "message": rng.choice(APPROVAL_ACTIONS[action]),
                "status": status,
                "comment": comment,
                "created_at": created_at,
                "decided_at": decided_at,
            }
        )
    return samples
