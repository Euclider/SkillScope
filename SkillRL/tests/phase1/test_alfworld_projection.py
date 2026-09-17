from agent_system.environments.env_package.alfworld.projection import alfworld_projection


def test_exact_action_only_response_is_valid() -> None:
    actions, valid = alfworld_projection(
        ["<action>go to drawer 1</action>\n"],
        [["go to drawer 1"]],
    )
    assert actions == ["go to drawer 1"]
    assert valid == [1]


def test_unwrapped_reasoning_without_think_block_remains_invalid() -> None:
    actions, valid = alfworld_projection(
        ["I should search first. <action>go to drawer 1</action>"],
        [["go to drawer 1"]],
    )
    assert actions == ["go to drawer 1"]
    assert valid == [0]
