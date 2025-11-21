# mybench/samples.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

@dataclass(frozen=True)
class SampleSpec:
    """하나의 xtc/pdb 샘플을 나타내는 정보."""
    name: str
    pdb_path: Path
    xtc_path: Path

def find_samples_in_dir(root: Path) -> List[SampleSpec]:
    """
    root 아래에서 xtc/pdb 쌍을 찾아 SampleSpec 리스트로 반환.

    bioemu-benchmarks 의 포맷을 그대로 따라갑니다.
    - topology.pdb + samples.xtc
    - 혹은 같은 stem 의 .pdb / .xtc
    """
    root = root.resolve()
    samples: List[SampleSpec] = []

    # 1) topology.pdb + samples.xtc 패턴
    for pdb_path in root.rglob("topology.pdb"):
        xtc_path = pdb_path.with_name("samples.xtc")
        if xtc_path.exists():
            name = pdb_path.parent.name
            samples.append(
                SampleSpec(
                    name=name,
                    pdb_path=pdb_path.resolve(),
                    xtc_path=xtc_path.resolve(),
                )
            )

    # 2) 동일 디렉터리 내 stem 매칭 패턴
    for xtc_path in root.rglob("*.xtc"):
        if xtc_path.name == "samples.xtc":
            # 위에서 이미 처리됨
            continue
        candidates = list(xtc_path.parent.glob(xtc_path.stem + ".pdb"))
        if not candidates:
            continue
        pdb_path = candidates[0]
        name = xtc_path.stem
        spec = SampleSpec(
            name=name,
            pdb_path=pdb_path.resolve(),
            xtc_path=xtc_path.resolve(),
        )
        if spec not in samples:
            samples.append(spec)

    return samples

def find_samples_in_dirs(roots: Iterable[Path]) -> List[SampleSpec]:
    """여러 sample_dirs 에서 샘플을 모읍니다."""
    all_samples: List[SampleSpec] = []
    for d in roots:
        all_samples.extend(find_samples_in_dir(Path(d)))

    # 중복 제거
    unique = {}
    for s in all_samples:
        key = (s.name, s.pdb_path, s.xtc_path)
        unique[key] = s
    return list(unique.values())