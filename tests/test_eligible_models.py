"""Tests for the allowlist of models the experiments may run."""

import pytest

from config.eligible_models import BLOCKED, ELIGIBLE, EXCLUDED, OVERRIDE_PHRASE, check_eligible, main


def test_no_eligible_model_matches_a_blocked_family():
    for name, checkpoint in ELIGIBLE.items():
        assert not BLOCKED.search(name), name
        assert not BLOCKED.search(checkpoint), checkpoint


def test_eligible_and_excluded_models_do_not_overlap():
    assert not set(ELIGIBLE) & set(EXCLUDED)
    assert not set(ELIGIBLE.values()) & {checkpoint for checkpoint, _ in EXCLUDED.values()}


def test_an_eligible_model_is_found_by_name_or_checkpoint():
    assert check_eligible("gpt-oss-20b") == "openai/gpt-oss-20b"
    assert check_eligible("openai/gpt-oss-20b") == "openai/gpt-oss-20b"
    assert check_eligible("ibm-granite/granite-3.3-8b-instruct") == "ibm-granite/granite-3.3-8b-instruct"


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen2.5-7B-Instruct",
        "OpenPipe/Qwen3-14B-Instruct",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "moonshotai/Kimi-K2-Instruct",
        "zai-org/GLM-4.5",
        "01-ai/Yi-1.5-34B",
        "baidu/ERNIE-4.5-21B",
        "tencent/Hunyuan-7B-Instruct",
        "my-lab/finetuned_qwen_model",
    ],
)
def test_models_from_blocked_families_are_rejected(model):
    with pytest.raises(ValueError, match="country of concern"):
        check_eligible(model)


def test_a_qwen_model_gets_a_warning_that_names_the_override():
    with pytest.raises(ValueError, match="WARNING.*Qwen model.*type the override") as error:
        check_eligible("Qwen/Qwen2.5-7B-Instruct")
    assert OVERRIDE_PHRASE in str(error.value)
    assert "Research Computing" in str(error.value)


def test_the_typed_override_lets_a_qwen_model_through_with_a_warning():
    with pytest.warns(UserWarning, match="typed Qwen override"):
        assert check_eligible("Qwen/Qwen2.5-7B-Instruct", OVERRIDE_PHRASE) == "Qwen/Qwen2.5-7B-Instruct"


@pytest.mark.parametrize("phrase", ["", "yes", "i accept running a qwen model from a country of concern", "override"])
def test_a_wrong_or_partial_phrase_is_refused(phrase):
    with pytest.raises(ValueError, match="override phrase does not match"):
        check_eligible("Qwen/Qwen2.5-7B-Instruct", phrase)


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "moonshotai/Kimi-K2-Instruct",
        "zai-org/GLM-4.5",
    ],
)
def test_the_override_does_not_open_other_blocked_families(model):
    with pytest.raises(ValueError, match="country of concern") as error:
        check_eligible(model, OVERRIDE_PHRASE)
    assert "override" not in str(error.value)


@pytest.mark.parametrize("model", ["llama-3.1-nemotron-nano-8B", "nemotron-3-nano-30b-a3b", "medgemma-27b-it"])
def test_the_override_does_not_open_excluded_models(model):
    with pytest.raises(ValueError, match="excluded"):
        check_eligible(model, OVERRIDE_PHRASE)


def test_the_override_changes_nothing_for_an_eligible_model():
    assert check_eligible("gpt-oss-20b", OVERRIDE_PHRASE) == "openai/gpt-oss-20b"


@pytest.mark.parametrize(
    "model, reason",
    [
        ("llama-3.1-nemotron-nano-8B", "Qwen"),
        ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "Qwen"),
        ("nemotron-3-nano-30b-a3b", "Qwen"),
        ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "teacher"),
        ("medgemma-27b-it", "medicine"),
    ],
)
def test_excluded_models_explain_why(model, reason):
    with pytest.raises(ValueError, match=f"excluded.*{reason}"):
        check_eligible(model)


def test_unknown_models_are_rejected():
    with pytest.raises(ValueError, match="not on the eligible list"):
        check_eligible("meta-llama/Llama-3.1-8B-Instruct")


def test_command_line_exit_codes(capsys):
    assert main(["gpt-oss-120b"]) == 0
    assert capsys.readouterr().out.strip() == "openai/gpt-oss-120b"
    assert main(["Qwen/Qwen2.5-7B-Instruct"]) == 1
    assert "country of concern" in capsys.readouterr().err
    assert main([]) == 1


def test_command_line_override(capsys):
    assert main(["Qwen/Qwen2.5-7B-Instruct"]) == 1
    assert "--override" in capsys.readouterr().err
    with pytest.warns(UserWarning, match="typed Qwen override"):
        assert main(["Qwen/Qwen2.5-7B-Instruct", "--override", OVERRIDE_PHRASE]) == 0
    assert capsys.readouterr().out.strip() == "Qwen/Qwen2.5-7B-Instruct"
    assert main(["Qwen/Qwen2.5-7B-Instruct", "--override", "yes"]) == 1
    assert main(["deepseek-ai/DeepSeek-R1", "--override", OVERRIDE_PHRASE]) == 1
    assert main(["Qwen/Qwen2.5-7B-Instruct", "--override"]) == 1
