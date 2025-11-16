"use client"

export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center p-4">
      <div className="relative w-8 h-8">
        <div className="absolute inset-0 rounded-full border-2 border-border"></div>
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-primary animate-spin"></div>
      </div>
    </div>
  )
}
