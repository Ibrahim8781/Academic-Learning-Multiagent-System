"use client"

import type { Agent } from "@/types"
import HealthStatus from "./health-status"
import { Clock } from "lucide-react"

interface AgentCardProps {
  agent: Agent
  health: "healthy" | "degraded" | "offline"
  isSelected: boolean
  onSelect: () => void
}

export default function AgentCard({ agent, health, isSelected, onSelect }: AgentCardProps) {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left p-3 rounded-md border transition-all focus-ring ${
        isSelected
          ? "bg-primary text-primary-foreground border-primary shadow-md"
          : "bg-background border-border hover:bg-muted hover:border-accent"
      }`}
      aria-pressed={isSelected}
      aria-label={`Select agent: ${agent.name}`}
    >
      <div className="space-y-2">
        {/* Header with name and health */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm truncate">{agent.name}</h3>
            <p className={`text-xs truncate ${isSelected ? "opacity-90" : "text-muted-foreground"}`}>
              {agent.description}
            </p>
          </div>
          <HealthStatus status={health} />
        </div>

        {/* Tags */}
        {agent.capabilities && agent.capabilities.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {agent.capabilities.slice(0, 2).map((cap) => (
              <span
                key={cap}
                className={`text-xs px-2 py-1 rounded-full ${
                  isSelected ? "bg-primary-foreground/20 text-primary-foreground" : "bg-accent/20 text-accent"
                }`}
              >
                {cap}
              </span>
            ))}
            {agent.capabilities.length > 2 && (
              <span className={`text-xs px-2 py-1 ${isSelected ? "text-primary-foreground" : "text-muted-foreground"}`}>
                +{agent.capabilities.length - 2}
              </span>
            )}
          </div>
        )}
      </div>
    </button>
  )
}
