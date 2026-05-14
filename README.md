## 本地安装
```
uv pip install -e . \
  --python /mnt/d/workspace/geesun_agent/.venv/bin/python

```

## run tests
```
# Linux 宿主机把证书先拷出来
sshpass -p opencloudos ssh -o StrictHostKeyChecking=no -p 10022 opencloudos@localhost "echo opencloudos | sudo -S cat /root/.local/share/mkcert/rootCA.pem" > ~/cube-ca.pem

export SSL_CERT_FILE=/home/dhp/projects/cube-cert/cube-ca.pem

uv run pytest tests/integration_tests/test_sandbox.py --asyncio-mode=auto

```