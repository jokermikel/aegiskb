"""pytest 全局夹具。

职责有两条：

1. **在导入 app.* 之前固定环境**：Settings 与 engine 都是导入期构造的，
   一旦先读到 .env 里的真实密钥，后面再改环境变量也没用。
2. **每次会话重建测试库**：演示数据集会随版本演进而变化（例如新增账号），
   残留的旧库会让断言莫名其妙地失败。这里用「清零同名文件」而不是删除文件，
   既避免了 Windows 上删除被占用的麻烦，效果同样是得到一个空的 SQLite 库。
"""
import os
import pathlib

TEST_DB = pathlib.Path(__file__).resolve().parents[1] / "test_aegiskb.db"

os.environ["DATABASE_URL"] = "sqlite:///test_aegiskb.db"
os.environ["JWT_SECRET"] = "test-secret"
# 覆盖 .env 中可能存在的真实百炼密钥，保证测试只走本地模式。
os.environ["DASHSCOPE_API_KEY"] = ""
# 固定离线向量后端：结果确定、不依赖网络、不消耗 token。
os.environ["EMBEDDING_PROVIDER"] = "offline"
os.environ["APP_ENV"] = "test"
os.environ["AUTO_CREATE_SCHEMA"] = "true"

for suffix in ("", "-journal", "-wal", "-shm"):
    stale = TEST_DB.with_name(TEST_DB.name + suffix)
    if stale.exists():
        stale.write_bytes(b"")
