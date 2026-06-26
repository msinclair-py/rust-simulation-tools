# Skills

Agent skills for working with `rust_simulation_tools`. Each subdirectory holds a
`SKILL.md` (YAML frontmatter + usage guide) that an AI coding assistant such as
[Claude Code](https://claude.com/claude-code) can load on demand to use the
package correctly.

| Skill | Covers |
|-------|--------|
| [`rust-simulation-tools`](rust-simulation-tools/SKILL.md) | Overview, install/build, loading topologies & trajectories; routes to the rest |
| [`rst-build-system`](rst-build-system/SKILL.md) | `SystemBuilder`: force fields, solvation, ions, write prmtop/inpcrd/PDB |
| [`rst-parameterize-ligand`](rst-parameterize-ligand/SKILL.md) | GAFF2 typing + AM1-BCC/Gasteiger charges; raw AM1 charges |
| [`rst-trajectory-analysis`](rst-trajectory-analysis/SKILL.md) | Selections, SASA, alignment, unwrapping, interaction fingerprints |
| [`rst-mmpbsa`](rst-mmpbsa/SKILL.md) | MM-PBSA / MM-GBSA binding free energy, decomposition, entropy |
| [`rst-minimize`](rst-minimize/SKILL.md) | Energy minimization with optional restraints |
| [`rst-ipsae`](rst-ipsae/SKILL.md) | ipSAE / pDockQ / LIS interface confidence scoring |

## Using with Claude Code

These are mirrored under `.claude/skills/` so they load automatically in this
repo. To use them in another project (or globally), copy the directories:

```bash
# project-scoped
cp -R skills/* /path/to/other-project/.claude/skills/
# or user-scoped (available everywhere)
cp -R skills/* ~/.claude/skills/
```

Each `SKILL.md` cross-references a runnable script in [`examples/`](../examples).
