# HTTPS 证书目录

[返回项目入口](../README.md) | [第四阶段范围](../4_https_split/README.md)

第四阶段使用 `certs/localhost/` 的独立本地凭据；第 1–3 阶段不使用证书。
生成命令与启动步骤见[第四阶段 README](../4_https_split/README.md)。

本目录除 README 外全部忽略版本控制。私钥不得公开；信任根、服务端证书、
主机名校验由第四阶段统一配置，不能关闭 TLS 校验来让测试通过。

| 文件 | 用途 |
|---|---|
| `localhost/ca.crt` | 客户端显式信任的开发 CA，不导入 Windows 全局信任 |
| `localhost/server.crt` | 服务端证书，SAN 包含 localhost 和 127.0.0.1，有效期 90 天 |
| `localhost/server.key` | 服务端私钥，保密 |
| `localhost/ca.key` | 开发 CA 私钥，保密，CA 有效期 365 天 |
| `localhost/token.txt` | 两端共用的随机访问令牌，保密 |
| 其他生成文件 | OpenSSL 本地配置、CSR、序列号，不需要手改 |

脚本将凭据目录 ACL 限制为当前用户。管理员仍可能读取，不防御同用户恶意程序。
不覆盖旧目录；轮换时停止两端，生成新的 certs 子目录，并同步修改
`4_https_split/config.json` 的 `https.credentials_dir`。不要混用不同代的 CA、证书和令牌。
