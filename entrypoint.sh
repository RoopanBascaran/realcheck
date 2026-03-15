#!/bin/bash
# Configure DNS at runtime (Docker overwrites /etc/resolv.conf at container start)
echo "nameserver 8.8.8.8" >> /etc/resolv.conf
echo "nameserver 8.8.4.4" >> /etc/resolv.conf

# Run gunicorn as non-root user
exec gosu user gunicorn app:app --bind 0.0.0.0:7860 --timeout 300 --workers 1 --log-level info --access-logfile - --error-logfile -
