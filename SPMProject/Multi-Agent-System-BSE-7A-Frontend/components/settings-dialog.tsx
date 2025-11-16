"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { useSettings } from "@/context/settings-context"

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { settings, updateSettings } = useSettings()
  const [localSettings, setLocalSettings] = useState(settings)

  const handleSave = () => {
    updateSettings(localSettings)
    onOpenChange(false)
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-card border border-border rounded-lg p-6 max-w-md w-full mx-4">
        <h2 className="text-lg font-semibold mb-4">Settings</h2>

        <div className="space-y-4">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={localSettings.enableLTM}
              onChange={(e) => setLocalSettings({ ...localSettings, enableLTM: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm">Enable Long-Term Memory (LTM)</span>
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={localSettings.enableSTM}
              onChange={(e) => setLocalSettings({ ...localSettings, enableSTM: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm">Enable Short-Term Memory (STM)</span>
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={localSettings.autoRoute}
              onChange={(e) => setLocalSettings({ ...localSettings, autoRoute: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm">Auto-route requests</span>
          </label>
        </div>

        <div className="flex gap-2 mt-6">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save</Button>
        </div>
      </div>
    </div>
  )
}
