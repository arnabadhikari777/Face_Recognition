from config import Config, load_dotenv_file


def test_defaults_do_not_expire_biometrics():
    assert Config().RETENTION_DAYS == 0


def test_env_overrides(tmp_path):
    c = Config.from_env({"RETENTION_DAYS": "30", "MAX_RECORDS": "10", "TRUST_PROXY": "true", "ENROLL_REQUIRE_ALL": "1"})
    assert c.RETENTION_DAYS == 30 and c.MAX_RECORDS == 10 and c.TRUST_PROXY and c.ENROLL_REQUIRE_ALL


def test_dotenv_does_not_override(tmp_path):
    f = tmp_path / ".env"; f.write_text("A=1\nB='two'\n#c\n")
    env = {"A": "keep"}
    assert load_dotenv_file(str(f), env) == 1 and env == {"A": "keep", "B": "two"}
