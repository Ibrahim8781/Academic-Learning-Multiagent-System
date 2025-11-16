"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface ErrorViewProps {
  title: string
  message: string
  rawError?: string
  onRetry?: () => void
  onCopyError?: () => void
}

export function ErrorView({ title, message, rawError, onRetry, onCopyError }: ErrorViewProps) {
  return (
    <Card className="p-6 bg-destructive/5 border-destructive/20">
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold text-destructive mb-2">{title}</h3>
          <p className="text-muted-foreground">{message}</p>
        </div>

        {rawError && (
          <div className="bg-background p-3 rounded-lg border border-border">
            <p className="text-xs font-mono text-muted-foreground whitespace-pre-wrap break-words">{rawError}</p>
          </div>
        )}

        <div className="flex gap-2">
          {onRetry && (
            <Button size="sm" onClick={onRetry}>
              Retry
            </Button>
          )}
          {onCopyError && (
            <Button size="sm" variant="outline" onClick={onCopyError}>
              Copy Error
            </Button>
          )}
        </div>
      </div>
    </Card>
  )
}
