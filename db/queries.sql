\echo '--- All saved messages ---'
SELECT id,
       left(session_id, 8) AS session,
       role,
       left(regexp_replace(content, '\s+', ' ', 'g'), 60) AS content,
       created_at
FROM chat_messages
ORDER BY id;

\echo ''
\echo '--- Messages per session ---'
SELECT left(session_id, 8) AS session,
       count(*)            AS messages,
       min(created_at)     AS started,
       max(created_at)     AS last_message
FROM chat_messages
GROUP BY session_id
ORDER BY started;
