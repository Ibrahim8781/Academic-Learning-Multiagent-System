"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface ErrorToastProps {
  message: string
  details?: string
  onRetry?: () => void
  onDismiss: () => void
}

export function ErrorToast({ message, details, onRetry, onDismiss }: ErrorToastProps) {
  const [isVisible, setIsVisible] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false)
      onDismiss()
    }, 5000)

    return () => clearTimeout(timer)
  }, [onDismiss])

  if (!isVisible) return null

  return (
    <Card className="fixed bottom-4 right-4 p-4 bg-destructive/10 border-destructive/20 max-w-md">
      <div className="space-y-2">
        <p className="font-semibold text-destructive">{message}</p>
        {details && <p className="text-sm text-muted-foreground">{details}</p>}
        <div className="flex gap-2">
          {onRetry && (
            <Button size="sm" variant="outline" onClick={onRetry}>
              Retry
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={onDismiss}>
            Dismiss
          </Button>
        </div>
      </div>
    </Card>
  )
}
