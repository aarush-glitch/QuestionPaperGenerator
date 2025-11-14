import json
from pathlib import Path

base = Path('data') / 'courses'
if not base.exists():
    print('No data/courses directory found.')
    raise SystemExit(1)

for course_dir in base.iterdir():
    if not course_dir.is_dir():
        continue
    cleaned = course_dir / 'cleaned_questions.json'
    if not cleaned.exists():
        print(f'No cleaned_questions.json for {course_dir.name}, skipping')
        continue
    print(f'Normalizing difficulty keys for {course_dir.name}')
    try:
        data = json.loads(cleaned.read_text(encoding='utf-8'))
        modified = False
        for g, items in data.items():
            for q in items:
                diff = q.get('difficulty') or q.get('difficulty_level')
                if diff and q.get('difficulty') != diff:
                    q['difficulty'] = diff
                    modified = True
                if diff and q.get('difficulty_level') != diff:
                    q['difficulty_level'] = diff
                    modified = True
        if modified:
            cleaned.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f'Updated {cleaned}')
        else:
            print(f'No changes for {cleaned}')
    except Exception as e:
        print(f'Failed to update {course_dir.name}: {e}')
