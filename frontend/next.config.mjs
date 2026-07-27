/** @type {import('next').NextConfig} */
const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  // 开发期把 /api/* 代理到 FastAPI —— 浏览器看来是同源,免 CORS。
  // 生产可改为 Nginx 反代,或前端独立部署 + 后端开 CORS。
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/:path*` }];
  },
};

export default nextConfig;
