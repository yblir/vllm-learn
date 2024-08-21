import vllm_module


def test_embedded_commit_defined():
    assert vllm_module.__commit__ != "COMMIT_HASH_PLACEHOLDER"
    # 7 characters is the length of a short commit hash
    assert len(vllm_module.__commit__) >= 7
