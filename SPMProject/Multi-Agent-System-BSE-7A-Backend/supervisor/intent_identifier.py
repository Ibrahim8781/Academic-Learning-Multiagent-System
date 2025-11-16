# supervisor/intent_identifier.py
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai

_logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    _logger.error("GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Agent descriptions for intent matching
AGENT_DESCRIPTIONS = {
    "adaptive_quiz_master_agent": {
        "name": "Adaptive Quiz Master Agent",
        "description": "Generates quizzes tailored to a learner's performance. Tracks past attempts and adapts difficulty/content for effective learning.",
        "keywords": ["quiz", "test", "questions", "assessment", "adaptive", "practice", "mcq", "true/false"],
        "example_intents": [
            "generate a quiz on Python",
            "create adaptive quiz for beginners",
            "make a test with 10 questions",
            "quiz me on linear algebra"
        ]
    },
    "assignment_coach_agent": {
        "name": "Assignment Coach Agent",
        "description": "Provides assignment understanding, task breakdown, resource suggestions, and progress guidance to students.",
        "keywords": ["assignment", "homework", "task", "breakdown", "guidance", "help with assignment", "project help"],
        "example_intents": [
            "help me with my assignment",
            "break down this task",
            "guide me through my homework",
            "assignment guidance needed"
        ]
    },
    "plagiarism_prevention_agent": {
        "name": "Plagiarism Prevention Agent",
        "description": "Analyzes submitted text for plagiarism and rephrases content to improve originality while preserving meaning.",
        "keywords": ["plagiarism", "check originality", "rephrase", "paraphrase", "originality", "copied content", "authenticity"],
        "example_intents": [
            "check my text for plagiarism",
            "rephrase this paragraph",
            "improve originality of my writing",
            "is this plagiarized"
        ]
    },
    "research_scout_agent": {
        "name": "Research Scout Agent",
        "description": "Helps students quickly find and summarize recent research papers, articles, or case studies related to any topic.",
        "keywords": ["research", "papers", "articles", "case studies", "literature review", "find papers", "academic sources"],
        "example_intents": [
            "find research papers on AI",
            "search for recent articles about machine learning",
            "get case studies on agile methodology",
            "literature review help"
        ]
    },
    "concept_reinforcement_agent": {
        "name": "Concept Reinforcement Agent",
        "description": "Generates short, focused learning activities for topics where learner's performance is weak or additional practice is needed.",
        "keywords": ["practice", "weak topics", "reinforcement", "struggling with", "need more practice", "flashcards", "review"],
        "example_intents": [
            "I need practice on data structures",
            "help me reinforce calculus concepts",
            "struggling with recursion",
            "create flashcards for biology"
        ]
    },
    "citation_manager_agent": {
        "name": "Citation Manager Agent",
        "description": "Automatically generates, validates, and manages academic citations in various formats (APA, MLA, Chicago, etc.).",
        "keywords": ["citation", "reference", "bibliography", "APA", "MLA", "Chicago", "cite", "format reference"],
        "example_intents": [
            "create citation in APA format",
            "format this reference",
            "generate bibliography",
            "cite this paper in MLA"
        ]
    },
    "peer_collaboration_agent": {
        "name": "Peer Collaboration Agent",
        "description": "Monitors team discussions and analyzes collaboration patterns, participation balance, and communication quality.",
        "keywords": ["team", "collaboration", "group work", "teamwork", "discussion analysis", "team feedback", "coordination"],
        "example_intents": [
            "analyze our team discussion",
            "feedback on group collaboration",
            "who's not contributing",
            "improve team communication"
        ]
    },
    "adaptive_flashcard_agent": {
        "name": "Adaptive Flashcard Agent",
        "description": "Dynamically generates and personalizes flashcards based on student's learning progress and performance.",
        "keywords": ["flashcard", "memorize", "recall", "study cards", "memory", "retention"],
        "example_intents": [
            "create flashcards for chemistry",
            "generate study cards",
            "help me memorize vocabulary",
            "adaptive flashcards for history"
        ]
    },
    "presentation_feedback_agent": {
        "name": "Presentation Feedback Agent",
        "description": "Analyzes presentation transcripts and provides feedback on confidence, material quality, pacing, and delivery.",
        "keywords": ["presentation", "feedback", "speech", "delivery", "public speaking", "transcript analysis", "presenting"],
        "example_intents": [
            "analyze my presentation",
            "feedback on my speech",
            "review my presentation transcript",
            "improve my delivery"
        ]
    },
    "lecture_insight_agent": {
        "name": "Lecture Insight Agent",
        "description": "Analyzes lecture audio to extract key insights, generate summaries, and suggest additional learning resources.",
        "keywords": ["lecture", "audio", "recording", "notes", "summarize lecture", "audio analysis", "class recording"],
        "example_intents": [
            "analyze this lecture recording",
            "summarize my class audio",
            "extract key points from lecture",
            "generate notes from recording"
        ]
    },
    "question_anticipator_agent": {
        "name": "Question Anticipator Agent",
        "description": "Predicts possible exam questions based on syllabus, past papers, and question patterns.",
        "keywords": ["exam questions", "predict questions", "syllabus analysis", "past papers", "exam preparation", "likely questions"],
        "example_intents": [
            "predict exam questions",
            "what questions might come",
            "analyze past papers",
            "anticipate exam topics"
        ]
    },
    "daily_revision_proctor_agent": {
        "name": "Daily Revision Proctor Agent",
        "description": "Monitors daily learning activities, generates personalized reminders, and provides adaptive study recommendations.",
        "keywords": ["revision", "daily study", "reminders", "study routine", "progress tracking", "study habits"],
        "example_intents": [
            "track my daily revision",
            "set study reminders",
            "monitor my progress",
            "help with study routine"
        ]
    },
    "study_scheduler_agent": {
        "name": "Study Scheduler Agent",
        "description": "Creates and optimizes personalized study timetables based on subjects, deadlines, and performance.",
        "keywords": ["schedule", "timetable", "study plan", "time management", "organize study", "calendar"],
        "example_intents": [
            "create study schedule",
            "make a timetable",
            "plan my study time",
            "organize my study sessions"
        ]
    },
    "exam_readiness_agent": {
        "name": "Exam Readiness Agent",
        "description": "Generates comprehensive assessments including MCQs, short questions, and long questions for exam preparation.",
        "keywords": ["exam", "assessment", "readiness", "mock test", "exam prep", "practice exam"],
        "example_intents": [
            "prepare for my exam",
            "create mock test",
            "assess my readiness",
            "generate practice exam"
        ]
    },
    "gemini-wrapper": {
        "name": "Gemini Wrapper (General LLM)",
        "description": "General-purpose text generation and conversation. Use when no specific agent matches or for general queries.",
        "keywords": ["general", "chat", "explain", "tell me about", "what is", "how does", "conversation"],
        "example_intents": [
            "explain quantum physics",
            "tell me about history",
            "general question",
            "chat about something"
        ]
    }
}

class IntentIdentifier:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def _build_agent_context(self) -> str:
        """Build a formatted string of all available agents and their capabilities."""
        context = "Available Learning System Agents:\n\n"
        for agent_id, info in AGENT_DESCRIPTIONS.items():
            context += f"Agent ID: {agent_id}\n"
            context += f"Name: {info['name']}\n"
            context += f"Description: {info['description']}\n"
            context += f"Example queries: {', '.join(info['example_intents'][:2])}\n\n"
        return context
    
    def _build_prompt(self, user_query: str, conversation_history: List[Dict] = None) -> str:
        """Build the prompt for Gemini to identify intent."""
        agent_context = self._build_agent_context()
        
        history_context = ""
        if conversation_history:
            history_context = "\nConversation History:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                history_context += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
        
        prompt = f"""You are an intelligent intent classifier for a learning management system. Your job is to analyze student queries and determine which specialized agent should handle the request.

{agent_context}

{history_context}

User Query: "{user_query}"

Analyze the query and respond with a JSON object in this EXACT format:
{{
    "agent_id": "the_agent_id_from_list_above",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this agent was chosen",
    "is_ambiguous": false,
    "clarifying_questions": [],
    "extracted_params": {{
        "topic": "extracted topic if applicable",
        "subject": "extracted subject if applicable",
        "difficulty": "extracted difficulty level if mentioned"
    }},
    "alternative_agents": []
}}

Important Rules:
1. Choose the MOST SPECIFIC agent that matches the query
2. Use "gemini-wrapper" ONLY if no specialized agent matches
3. Set "is_ambiguous" to true if you need more information
4. If ambiguous, provide 2-3 clarifying questions
5. Set confidence between 0.0 and 1.0
6. If multiple agents could handle it, list them in "alternative_agents"
7. Extract any relevant parameters from the query (topic, subject, difficulty, etc.)

Respond ONLY with the JSON object, no additional text."""

        return prompt
    
    async def identify_intent(
        self, 
        user_query: str, 
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Identify the intent and appropriate agent for a user query.
        
        Returns:
            Dict with agent_id, confidence, reasoning, and other metadata
        """
        try:
            prompt = self._build_prompt(user_query, conversation_history)
            
            _logger.info(f"Identifying intent for query: {user_query}")
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON response
            intent_result = json.loads(response_text)
            
            # Validate agent_id exists
            agent_id = intent_result.get("agent_id")
            if agent_id not in AGENT_DESCRIPTIONS:
                _logger.warning(f"LLM returned unknown agent_id: {agent_id}, defaulting to gemini-wrapper")
                intent_result["agent_id"] = "gemini-wrapper"
                intent_result["confidence"] = 0.5
            
            _logger.info(f"Intent identified: {intent_result.get('agent_id')} (confidence: {intent_result.get('confidence')})")
            
            return intent_result
            
        except json.JSONDecodeError as e:
            _logger.error(f"Failed to parse LLM response as JSON: {e}")
            _logger.error(f"Raw response: {response_text}")
            return self._fallback_intent(user_query)
            
        except Exception as e:
            _logger.error(f"Error in intent identification: {e}")
            return self._fallback_intent(user_query)
    
    def _fallback_intent(self, user_query: str) -> Dict:
        """Fallback when LLM fails - use keyword matching."""
        _logger.warning("Using fallback keyword-based intent identification")
        
        query_lower = user_query.lower()
        best_match = None
        best_score = 0
        
        for agent_id, info in AGENT_DESCRIPTIONS.items():
            score = sum(1 for keyword in info['keywords'] if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_match = agent_id
        
        if best_match and best_score > 0:
            return {
                "agent_id": best_match,
                "confidence": min(0.7, best_score * 0.2),
                "reasoning": "Fallback keyword matching used",
                "is_ambiguous": False,
                "clarifying_questions": [],
                "extracted_params": {},
                "alternative_agents": []
            }
        
        # Ultimate fallback to gemini-wrapper
        return {
            "agent_id": "gemini-wrapper",
            "confidence": 0.3,
            "reasoning": "No specific agent matched, using general LLM",
            "is_ambiguous": False,
            "clarifying_questions": [],
            "extracted_params": {},
            "alternative_agents": []
        }

# Global instance
_intent_identifier = None

def get_intent_identifier() -> IntentIdentifier:
    """Get or create the global intent identifier instance."""
    global _intent_identifier
    if _intent_identifier is None:
        _intent_identifier = IntentIdentifier()
    return _intent_identifier