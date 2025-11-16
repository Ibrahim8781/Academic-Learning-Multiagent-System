"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface OfflineWarningProps {
  agentName: string
  onSelectAlternative?: () => void
}

export function OfflineWarning({ agentName, onSelectAlternative }: OfflineWarningProps) {
  return (
    <Card className="p-4 bg-yellow-50 dark:bg-yellow-950 border-yellow-200 dark:border-yellow-800">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-semibold text-yellow-900 dark:text-yellow-100">Agent Offline</p>
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            {agentName} is currently offline. Try another agent or request offline handling.
          </p>
        </div>
        {onSelectAlternative && (
          <Button size="sm" variant="outline" onClick={onSelectAlternative}>
            Select Alternative
          </Button>
        )}
      </div>
    </Card>
  )
}
