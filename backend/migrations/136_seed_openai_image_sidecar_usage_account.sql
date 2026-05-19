-- Seed a virtual account used only to satisfy usage_logs.account_id for
-- OpenAI-compatible image sidecar billing records. It is disabled and
-- unschedulable, so it can never be selected as an upstream account.

INSERT INTO accounts (
    id,
    name,
    platform,
    type,
    credentials,
    extra,
    concurrency,
    priority,
    status,
    schedulable,
    rate_multiplier,
    created_at,
    updated_at
)
VALUES (
    0,
    'openai-image-sidecar',
    'openai',
    'upstream',
    '{}'::jsonb,
    '{"virtual": true, "purpose": "openai_image_sidecar_usage"}'::jsonb,
    0,
    100,
    'disabled',
    false,
    1.0,
    NOW(),
    NOW()
)
ON CONFLICT (id) DO NOTHING;

SELECT setval(
    pg_get_serial_sequence('accounts', 'id'),
    GREATEST(COALESCE((SELECT MAX(id) FROM accounts), 1), 1),
    true
);
