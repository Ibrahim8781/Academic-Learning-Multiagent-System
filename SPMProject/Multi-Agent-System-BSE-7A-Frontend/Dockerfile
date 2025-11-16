# Dockerfile for Next.js frontend

# 1. Base image for installing dependencies
FROM node:20-alpine AS deps
# Install pnpm
RUN npm install -g pnpm

WORKDIR /app
# Copy package.json and pnpm-lock.yaml to install dependencies
COPY package.json pnpm-lock.yaml ./
# Install dependencies using pnpm
RUN pnpm install

# 2. Base image for building the application
FROM node:20-alpine AS builder
WORKDIR /app
# Copy dependencies from the 'deps' stage
COPY --from=deps /app/node_modules ./node_modules
# Copy the rest of the application source code
COPY . .
# Install pnpm globally to run the build command
RUN npm install -g pnpm
# Build the Next.js application
RUN pnpm build

# 3. Final image for running the application
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# Install pnpm globally to run the start command
RUN npm install -g pnpm

# Copy necessary files from the builder stage
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.mjs ./
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Expose the port the app runs on
EXPOSE 3000

# The command to start the app
CMD ["pnpm", "start"]
