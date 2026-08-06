# 前端生产镜像:多阶段构建 —— 编译在一个阶段,运行镜像只带产物,不带源码和构建工具。
FROM node:20-alpine AS deps
WORKDIR /app
# 只先拷依赖清单:清单没变就命中缓存,改业务代码不必重装依赖
COPY package.json package-lock.json* ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# 关掉遥测,构建更快也更干净
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
# 不用 root 跑:容器被攻破时,攻击者拿到的也只是个无权限用户
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001
# standalone 产物已自带运行所需的最小依赖
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
