---
title: Engineering Handbook
source: Engineering Handbook
url: https://intranet.example.com/eng/handbook
author: Platform Team
date: 2025-06-02
---

# Engineering Handbook

## Payments Service

### Configuration

Set the timeout to 30 seconds. Anything longer will trigger the circuit
breaker and the request will be retried against the secondary region.

The service reads its configuration from environment variables at boot. A
configuration change therefore requires a rolling restart; there is no hot
reload.

```yaml
payments:
  timeout_seconds: 30
  retries: 3
  circuit_breaker:
    failure_threshold: 5
    reset_seconds: 60
```

### Database Migrations

Migrations run automatically on deploy. To run one by hand, first take the
service out of the load balancer pool.

Then run the migration script. If it fails, check the logs in /var/log/db and
retry with --force. Never run --force against production without a fresh
snapshot; the flag skips the pre-flight consistency check.

## Search Service

### Indexing

The search index rebuilds nightly at 02:00 UTC. A full rebuild takes about
40 minutes for the current corpus size. Incremental updates are applied every
five minutes from the change log.

If the nightly rebuild fails, the previous index remains live and an alert is
raised in the platform channel. Search stays available on stale data rather
than going down.

### On-call

The search on-call rotation is weekly, handing over on Monday at 10:00 local
time. Page severity 1 for total search unavailability, severity 2 for
degraded relevance or stale indexes.
