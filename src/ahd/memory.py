"""Training-only retrieval from measured edits in the current EvoX run."""
import ast
from collections import Counter

MEMORY_BUDGET = 384
MEMORY_MODES = {'evox_lineage_memory': 'summary', 'evox_lineage_reward': 'summary'}


def changed_functions(before, after):
    def functions(source):
        return {node.name: ast.dump(node) for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    left, right = functions(before), functions(after)
    return sorted(name for name in left.keys() | right.keys() if left.get(name) != right.get(name))


def related(edit, parent):
    return bool({edit['parent_id'], edit['child_id']} & {parent.id, parent.parent_id})


def retrieve(history, parent, operator, profile, mode):
    """Return a compact summary and independently removable evidence sections."""
    if not history:
        return [], []
    weakest = sorted(range(len(profile)), key=profile.__getitem__)[:max(1, len(profile)//3)]
    candidates = [row for row in history if row['valid'] and row['gain'] > 0]
    failures = [row for row in history if not row['valid']]
    selected = []
    ranking = []

    def rank(row):
        delta = row['case_deltas']
        overlap = sum(delta[i] > 0 for i in weakest) if len(profile) > 1 and len(delta) == len(profile) else 0
        redundancy = sum(bool(set(row['changed']) & set(other['changed']))
                         or row['parent_id'] == other['parent_id'] for other in selected)
        return overlap, int(related(row, parent))-redundancy, -redundancy, row['operator'] == operator, row['gain'], row['id']

    limit = 1 if mode == 'one' else 3
    if mode == 'recent':
        selected = sorted(candidates + failures, key=lambda row: row['id'], reverse=True)[:limit]
        ranking = [dict(id=row['id'], reason='recent') for row in selected]
    else:
        relevant_failures = [row for row in failures if related(row, parent) or row['operator'] == operator]
        success_limit = limit - int(limit > 1 and bool(relevant_failures))
        while candidates and len(selected) < success_limit:
            distinct = [row for row in candidates if not any(
                row['changed'] == old['changed'] and row['parent_id'] == old['parent_id']
                for old in selected)]
            if not distinct:
                break
            row = max(distinct, key=rank)
            ranking.append(dict(id=row['id'], reason='relevance', rank=list(rank(row))))
            candidates.remove(row)
            selected.append(row)
        if len(selected) < limit and failures:
            row = max(relevant_failures or failures, key=rank)
            ranking.append(dict(id=row['id'], reason='failure', rank=list(rank(row))))
            selected.append(row)

    sections = []
    if mode in ('summary', 'recent', 'actionable', 'delta'):
        lineage = [row for row in history if related(row, parent)][-8:]
        if lineage:
            issues = Counter(row['issue'] for row in lineage if row['issue'])
            summary = (f'Linked recent edits: {len(lineage)}; valid improvements: '
                       f'{sum(row["valid"] and row["gain"] > 0 for row in lineage)}.')
            best = max((row for row in lineage if row['valid'] and row['gain'] > 0),
                       key=lambda row: row['gain'], default=None)
            if best:
                summary += f' Best linked improvement: edit {best["id"]}, delta {best["gain"]:+.5g}.'
            if issues:
                summary += f' Most frequent failure: {issues.most_common(1)[0][0]}.'
            sections.append(('summary', summary))
    for row in selected:
        changed = ', '.join(name[:40] for name in row['changed'][:4]) or 'module-level edit'
        text = (f'Edit {row["id"]}: {changed}. Training delta {row["gain"]:+.5g}; '
                f'valid={row["valid"]}; runtime={row["runtime"]:.3g}s.')
        changes = sorted(enumerate(row['case_deltas'], 1), key=lambda pair: abs(pair[1]), reverse=True)[:3]
        if mode == 'actionable':
            deltas = list(enumerate(row['case_deltas'], 1))
            gain = max((pair for pair in deltas if pair[1] > 0),
                       key=lambda pair: pair[1], default=None)
            regression = min((pair for pair in deltas if pair[1] < 0),
                             key=lambda pair: pair[1], default=None)
            if gain:
                text += f' Largest gain: case {gain[0]} ({gain[1]:+.4g}).'
            if regression:
                text += f' Largest regression: case {regression[0]} ({regression[1]:+.4g}).'
            if row['valid']:
                text += ' Use this trade-off to decide which part of the edit to carry forward.'
            else:
                text += ' Avoid repeating the observed failure.'
        elif changes:
            text += ' Case deltas: ' + ', '.join(f'{i}={delta:+.4g}' for i, delta in changes) + '.'
        if mode == 'delta' and row.get('code_delta'):
            text += ' Code excerpt: ' + row['code_delta']
        if row['issue']:
            text += f' Observed failure: {row["issue"]}.'
        sections.append((str(row['id']), text))
    return sections, ranking
