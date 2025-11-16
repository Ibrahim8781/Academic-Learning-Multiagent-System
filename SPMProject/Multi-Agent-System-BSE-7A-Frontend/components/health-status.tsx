"use client"

import { Circle } from "lucide-react"

interface HealthStatusProps {
  status: "healthy" | "degraded" | "offline"
}

export default function HealthStatus({ status }: HealthStatusProps) {
  const statusConfig = {
    healthy: {
      color: "text-green-600 dark:text-green-400",
      bgColor: "bg-green-100 dark:bg-green-900/30",
      label: "Healthy",
    },
    degraded: {
      color: "text-yellow-600 dark:text-yellow-400",
      bgColor: "bg-yellow-100 dark:bg-yellow-900/30",
      label: "Degraded",
    },
    offline: {
      color: "text-red-600 dark:text-red-400",
      bgColor: "bg-red-100 dark:bg-red-900/30",
      label: "Offline",
    },
  }

  const config = statusConfig[status]

  return (
    <div
      className={`flex items-center gap-1.5 px-2 py-1 rounded-full ${config.bgColor}`}
      role="status"
      aria-label={`Agent status: ${config.label}`}
    >
      <Circle className={`w-2 h-2 fill-current ${config.color}`} />
      <span className={`text-xs font-medium ${config.color}`}>{config.label}</span>
    </div>
  )
}
