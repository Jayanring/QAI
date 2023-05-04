# 使用官方 Rust 镜像的最新稳定版作为基础镜像
FROM rust:slim-bullseye AS builder

# 设置工作目录
WORKDIR /usr/src/myrustapp

# 复制项目中的所有文件到工作目录
COPY . .

# 编译项目
RUN cargo build --release

# 使用一个轻量级的基础镜像，用于运行已编译的程序
FROM debian:buster-slim

# 在新的镜像中创建一个运行用户
RUN useradd -ms /bin/bash myrustappuser

# 切换到新创建的用户
USER myrustappuser

# 设置工作目录
WORKDIR /home/myrustappuser

# 从构建阶段复制编译好的可执行文件到运行阶段的镜像
COPY --from=builder /usr/src/myrustapp/target/release/qai .

# 复制依赖的动态库 (.so 文件)
COPY --from=builder /usr/src/myrustapp/libpdfium.so .

# 复制环境变量配置文件 (.env)
COPY --from=builder /usr/src/myrustapp/.env .

# 暴露 8080 端口
EXPOSE 8080

# 运行 Rust 应用程序
CMD ["./qai"]
