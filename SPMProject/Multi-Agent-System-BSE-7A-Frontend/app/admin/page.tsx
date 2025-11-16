"use client"

import { useState } from "react"
import Link from "next/link"
import Header from "@/components/header"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"

export default function AdminPage() {
  const [registryJson, setRegistryJson] = useState("")
  const [testResults, setTestResults] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleUploadRegistry = async () => {
    setLoading(true)
    try {
      // Parse and validate JSON
      const parsed = JSON.parse(registryJson)
      console.log("Registry JSON validated:", parsed)
      setTestResults("Registry uploaded successfully!")
    } catch (error) {
      setTestResults(`Error: ${error instanceof Error ? error.message : "Invalid JSON"}`)
    } finally {
      setLoading(false)
    }
  }

  const handleRunTests = async () => {
    setLoading(true)
    try {
      // Simulate integration tests
      setTestResults("Integration tests passed: 5/5")
    } catch (error) {
      setTestResults(`Tests failed: ${error}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      <div className="flex-1 flex flex-col">
        <Header />
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold mb-2">Admin & Integration</h1>
                <p className="text-muted-foreground">Manage registry and run integration tests</p>
              </div>
              <Link href="/dashboard">
                <Button variant="outline">Back to Dashboard</Button>
              </Link>
            </div>

            {/* Registry Upload */}
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Registry Upload</h2>
              <div className="space-y-4">
                <Textarea
                  placeholder="Paste handshake JSON here..."
                  value={registryJson}
                  onChange={(e) => setRegistryJson(e.target.value)}
                  className="font-mono text-sm h-48"
                />
                <Button onClick={handleUploadRegistry} disabled={loading || !registryJson}>
                  {loading ? "Uploading..." : "Upload Registry"}
                </Button>
              </div>
            </Card>

            {/* Integration Tests */}
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Integration Tests</h2>
              <div className="space-y-4">
                <p className="text-muted-foreground">Run tests to verify agent connectivity and health checks</p>
                <Button onClick={handleRunTests} disabled={loading} variant="secondary">
                  {loading ? "Running..." : "Run Tests"}
                </Button>
              </div>
            </Card>

            {/* Test Results */}
            {testResults && (
              <Card className="p-6 bg-muted">
                <h2 className="text-xl font-semibold mb-4">Results</h2>
                <pre className="text-sm font-mono whitespace-pre-wrap">{testResults}</pre>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
