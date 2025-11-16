import { type Message } from "@/types"
import { Bot, User, AlertTriangle } from "lucide-react"

interface ChatMessageProps {
  message: Message & { metadata?: any }
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.type === "user"
  const isError = message.type === "error"

  const renderContent = () => {
    if (typeof message.content === "string") {
      return <p className="text-sm whitespace-pre-wrap">{message.content}</p>
    }
    return <p className="text-sm whitespace-pre-wrap">{JSON.stringify(message.content)}</p>
  }

  const Icon = isUser ? User : isError ? AlertTriangle : Bot

  return (
    <div className={`flex items-start gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div
          className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center ${
            isError ? "bg-destructive/20 text-destructive" : "bg-accent text-accent-foreground"
          }`}
        >
          <Icon className="w-5 h-5" />
        </div>
      )}
      <div
        className={`max-w-xs lg:max-w-2xl px-4 py-2 rounded-lg ${
          isError
            ? "bg-destructive/10 border border-destructive/20 text-destructive-foreground"
            : isUser
            ? "bg-primary text-primary-foreground"
            : "bg-accent text-accent-foreground"
        }`}
      >
        {renderContent()}
        {message.metadata && (
          <div className="mt-2 pt-2 border-t border-white/20 text-xs opacity-80">
            <p>Execution Time: {message.metadata.executionTime}ms</p>
            <p>Trace: {message.metadata.agentTrace.join(" -> ")}</p>
          </div>
        )}
        <p className="text-xs opacity-70 mt-1 text-right">
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-primary/20 text-primary">
          <Icon className="w-5 h-5" />
        </div>
      )}
    </div>
  )
}
