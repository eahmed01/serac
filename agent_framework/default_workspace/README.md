# Sample Workspace

A small, self-contained project workspace for agent tool execution.
Mounted read-only in the Docker sandbox at `/repo`.

## Structure

```
src/           — Python modules (read-only)
data/          — Sample data files (read-only)
config/        — Configuration files (read-only)
```

## Write Target

All write operations go to `/tmp/workspace` (writable tmpfs).

## Rules

- You may read any file under `/repo`
- You may write only to `/tmp/workspace`
- No network access is available
