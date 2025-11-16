# Education & Learning Agents Hub UI

A production-ready React single-page application for interacting with a multi-agent system. Built with TypeScript, Tailwind CSS, and modern React patterns.

## Features

- **Agent Discovery**: Browse available agents with real-time health status
- **Interactive Chat**: Send requests and receive responses from agents
- **History Management**: View and manage conversation history per agent
- **Settings**: Configure LTM/STM and auto-routing preferences
- **Error Handling**: Graceful error boundaries and offline support
- **Responsive Design**: Mobile-first, accessible UI (WCAG AA)

## Project Structure

```
src/
├── app/
│   ├── page.tsx           # Main app component
│   └── globals.css        # Global styles and theme
├── components/
│   ├── header.tsx         # App header
│   ├── agent-list.tsx     # Agent list display
│   ├── agent-card.tsx     # Individual agent card
│   ├── chat-window.tsx    # Chat interface
│   ├── request-composer.tsx # Input form
│   ├── chat-message.tsx   # Message display
│   ├── history-panel.tsx  # Conversation history
│   ├── health-status.tsx  # Agent health indicator
│   ├── settings-dialog.tsx # Settings modal
│   └── error-boundary.tsx # Error handling
├── context/
│   ├── agent-context.tsx  # Agent state management
│   ├── history-context.tsx # History state management
│   └── settings-context.tsx # Settings state management
├── lib/
│   └── api-service.ts     # Backend API wrapper
└── types/
    └── agent.ts           # TypeScript interfaces
```

## Backend API Contract

### GET /supervisor/registry
Fetch available agents.

**Response:**
```json
{
  "agents": [
    {
      "id": "agent-1",
      "name": "Research Agent",
      "description": "Searches and summarizes information",
      "capabilities": ["search", "summarize"]
    }
  ]
}
```

### POST /supervisor/request
Submit a request to an agent.

**Request:**
```json
{
  "agentId": "agent-1",
  "request": "What is AI?",
  "autoRoute": false
}
```

**Response:**
```json
{
  "response": "AI is artificial intelligence...",
  "agentId": "agent-1",
  "timestamp": "2024-10-24T10:30:00Z"
}
```

### GET /agent/{id}/health
Check agent health status.

**Response:**
```json
{
  "status": "healthy"
}
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repo-url>

# Install dependencies
npm install

# Set environment variables
echo "NEXT_PUBLIC_API_URL=http://localhost:3001" > .env.local
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## Testing

```bash
npm test
```

## Configuration

### Environment Variables

- `NEXT_PUBLIC_API_URL`: Backend API base URL (default: `http://localhost:3001`)

### Settings

Users can configure:
- **Enable LTM**: Store long-term conversation memory
- **Enable STM**: Store short-term conversation memory
- **Auto-route**: Automatically route requests to appropriate agents

## Accessibility

- WCAG AA compliant
- Semantic HTML structure
- ARIA labels and roles
- Keyboard navigation support
- Screen reader friendly

## Performance

- Client-side state management with React Context
- LocalStorage for history and settings persistence
- Efficient re-renders with proper memoization
- Lazy loading of components

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## License

MIT
