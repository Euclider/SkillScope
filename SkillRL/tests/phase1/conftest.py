import json

import pytest


@pytest.fixture
def skill_bank(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(json.dumps({
        "general_skills": [
            {"skill_id": "gen_001", "title": "Explore", "principle": "Explore once", "when_to_apply": "always"},
        ],
        "task_specific_skills": {
            "clean": [
                {"skill_id": "cle_001", "title": "Clean", "principle": "Use sink", "when_to_apply": "clean tasks"},
            ],
        },
        "common_mistakes": [
            {"mistake_id": "err_001", "description": "Loop", "how_to_avoid": "Change action"},
        ],
    }), encoding="utf-8")
    return path

