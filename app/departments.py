"""部门取值范围与校验。

这是演示与默认的部门集合；生产环境应从企业目录服务（OIDC/LDAP）同步，
而不是依赖这份常量。
"""
from __future__ import annotations

# 全员可见的保留取值。
ALL_DEPARTMENT = "all"
# 子树通配后缀，例如 engineering/* 表示工程部及其子部门。
SUBTREE_SUFFIX = "/*"

# 与 app/seed_data.py 的演示账号保持一致，测试会断言两者不会漂移。
KNOWN_DEPARTMENTS = (
    "engineering",
    "finance",
    "hr",
    "it",
    "legal",
    "ops",
    "sales",
    "security",
)


def validate_department_selectors(value: str) -> str:
    """校验并规范化部门写法，非法取值抛 ValueError。

    允许的写法：
    - ``all``                  全员可见
    - ``engineering``          单个已知部门
    - ``engineering,it``       多部门（逗号分隔）
    - ``engineering/*``        已知部门及其子树

    必须校验的原因：把部门名拼错（例如 ``enginering``）时，文档会入库成功，
    但没有任何账号能检索到它 —— 属于最难排查的一类静默故障。
    """
    selectors: list[str] = []
    for item in (value or "").split(","):
        selector = item.strip().lower()
        if not selector:
            continue
        base = selector[: -len(SUBTREE_SUFFIX)] if selector.endswith(SUBTREE_SUFFIX) else selector
        if selector != ALL_DEPARTMENT and base not in KNOWN_DEPARTMENTS:
            allowed = "、".join(KNOWN_DEPARTMENTS)
            raise ValueError(
                f"未知部门 {selector!r}。可用取值：all、{allowed}；"
                "同时支持逗号分隔的多部门与 `部门/*` 子树写法。"
            )
        selectors.append(selector)
    if not selectors:
        raise ValueError("department 不能为空")
    # 去重但保留顺序，避免 "it,it" 这类冗余写法落库。
    return ",".join(dict.fromkeys(selectors))
