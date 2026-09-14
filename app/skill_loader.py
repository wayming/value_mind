"""SKILL.md 加载器:解析 YAML frontmatter + 正文。

skills 目录布局:skills/<skill_id>/SKILL.md。
每个 skill 的正文只注入它自己所在节点的系统提示词——按 skill 切分小步,
不把所有内容一次性塞给 LLM。
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillDoc:
    id: str
    name: str
    description: str
    body: str          # 正文(分步分析指导,作为该节点 LLM 的系统提示词)
    path: Path


def load_skill(skill_id: str) -> SkillDoc:
    path = SKILLS_DIR / skill_id / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"skill 不存在: {path}")
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path}: 缺少 YAML frontmatter(--- ... ---)")
    meta = yaml.safe_load(m.group(1)) or {}
    return SkillDoc(
        id=skill_id,
        name=meta.get("name", skill_id),
        description=str(meta.get("description", "")),
        body=text[m.end():].strip(),
        path=path,
    )


def list_skill_ids() -> list[str]:
    """按目录名排序的所有 skill id。"""
    if not SKILLS_DIR.exists():
        return []
    return sorted(d.name for d in SKILLS_DIR.iterdir() if (d / "SKILL.md").exists())
