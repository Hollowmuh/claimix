import os
import json
import time
from typing import Dict, List
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

from triage_agent import run_triage
from attachment_details import generate_attachment_details
from clarification_call import run_clarifying_question
from utils import load_json, save_json, get_session_folder, load_claim_state, save_claim_state

load_dotenv()
SESSIONS_DIR = "sessions"
CLAIM_FILE = "claim.json"
CONTEXT_FILE = "context.json"
FOLLOW_UP_FILE = "follow_up.json"
ATTACHMENT_DATA_FILE = "attachment_data.json"

# Map incident types to their corresponding agents
INCIDENT_TYPE_TO_AGENT = {
    # MODULE 1 – Physical Loss & Damage
    "accidental_and_glass_damage": "accidental_and_glass_assistant",
    "fire": "fire_assistant",
    "theft": "theft_assistant",
    "ancillary_property": "ancillary_assistant",

    # MODULE 2 – Third-Party Liability
    "third_party_injury": "third_party_injury_assistant",
    "third_party_property": "third_party_property_assistant",
    "special_liability": "special_liability_assistant",
    "legal_and_statutory": "legal_and_statutory_assistant",

    # MODULE 3 – Personal Protection
    "personal_injury": "personal_injury_assistant",
    "personal_convenience": "personal_convenience_assistant",
    "personal_property": "personal_property_assistant",

    # MODULE 4 – Policy Governance
    "territorial_usage": "territorial_and_usage_assistant",
    "general_exceptions": "general_exceptions_assistant",
    "vehicle_security": "vehicle_security_assistant",
    "administrative": "adminstrative_assistant"
}

# Assistant IDs for each agent - replace with your actual assistant IDs
ASSISTANT_IDS = {
    # MODULE 1 – Physical Loss & Damage
    "accidental_and_glass_assistant": os.getenv("ACCIDENTAL_AND_GLASS_ASSISTANT_ID"),
    "fire_assistant": os.getenv("FIRE_ASSISTANT_ID"),
    "theft_assistant": os.getenv("THEFT_ASSISTANT_ID"),
    "ancillary_assistant": os.getenv("ANCILLARY_ASSISTANT_ID"),

    # MODULE 2 – Third-Party Liability
    "third_party_injury_assistant": os.getenv("THIRD_PARTY_INJURY_ASSISTANT_ID"),
    "third_party_property_assistant": os.getenv("THIRD_PARTY_PROPERTY_ASSISTANT_ID"),
    "special_liability_assistant": os.getenv("SPECIAL_LIABILITY_ASSISTANT_ID"),
    "legal_and_statutory_assistant": os.getenv("LEGAL_AND_STATUTORY_ASSISTANT_ID"),

    # MODULE 3 – Personal Protection
    "personal_injury_assistant": os.getenv("PERSONAL_INJURY_ASSISTANT_ID"),
    "personal_convenience_assistant": os.getenv("PERSONAL_CONVENIENCE_ASSISTANT_ID"),
    "personal_property_assistant": os.getenv("PERSONAL_PROPERTY_ASSISTANT_ID"),

    # MODULE 4 – Policy Governance
    "territorial_and_usage_assistant": os.getenv("TERRITORIAL_AND_USAGE_ASSISTANT_ID"),
    "general_exceptions_assistant": os.getenv("GENERAL_EXCEPTIONS_ASSISTANT_ID"),
    "vehicle_security_assistant": os.getenv("VEHICLE_SECURITY_ASSISTANT_ID"),
    "adminstrative_assistant": os.getenv("ADMINISTRATIVE_ASSISTANT_ID"),
}


class Orchestrator:
    def __init__(self):
        self.sessions_dir = SESSIONS_DIR
        self.incident_type_to_agent = INCIDENT_TYPE_TO_AGENT
        self.assistant_ids = ASSISTANT_IDS
        self.client = OpenAI()  # Initialize OpenAI client
        
    def init_context(self, email: str):
        """Initialize conversation context"""
        folder = get_session_folder(email)
        context_path = os.path.join(folder, CONTEXT_FILE)
        
        if not os.path.exists(context_path):
            default_context = {
                "conversation_history": [],
                "attachment_details": {},
                "last_updated": time.time()
            }
            save_json(context_path, default_context)

    def load_attachment_data(self, email: str) -> Dict:
        """Load attachment data from attachment_data.json"""
        folder = get_session_folder(email)
        attachment_path = os.path.join(folder, ATTACHMENT_DATA_FILE)
        
        if os.path.exists(attachment_path):
            return load_json(attachment_path)
        return {}

    def update_context(self, email: str, user_message: str, attachments: List[str] = None):
        """Update conversation context with new user message and attachment details"""
        folder = get_session_folder(email)
        context_path = os.path.join(folder, CONTEXT_FILE)
        
        # Load existing context
        if os.path.exists(context_path):
            context = load_json(context_path)
        else:
            context = {
                "conversation_history": [],
                "attachment_details": {},
                "last_updated": time.time()
            }
        
        # Add user message to conversation history
        if user_message.strip():
            context["conversation_history"].append({
                "role": "user",
                "content": user_message,
                "timestamp": time.time()
            })
        
        # Handle attachments
        if attachments:
            context["conversation_history"].append({
                "role": "user",
                "content": f"[User uploaded {len(attachments)} attachment(s)]",
                "timestamp": time.time(),
                "attachments": attachments
            })
        
        # Load and merge attachment details
        attachment_data = self.load_attachment_data(email)
        if attachment_data:
            context["attachment_details"] = attachment_data
        
        context["last_updated"] = time.time()
        save_json(context_path, context)

    def get_conversation_context(self, email: str) -> Dict:
        """Get comprehensive conversation context"""
        folder = get_session_folder(email)
        context_path = os.path.join(folder, CONTEXT_FILE)
        
        if os.path.exists(context_path):
            return load_json(context_path)
        return {
            "conversation_history": [],
            "attachment_details": {},
            "last_updated": time.time()
        }

    def save_agent_message(self, email: str, agent_name: str, message: str, role: str = "assistant"):
        """Save agent message to agent-specific messages file"""
        folder = get_session_folder(email)
        agent_messages_path = os.path.join(folder, f"{agent_name}_messages.json")
        
        if os.path.exists(agent_messages_path):
            messages = load_json(agent_messages_path)
        else:
            messages = []
        
        msg_entry = {
            "role": role,
            "content": message,
            "timestamp": time.time()
        }
        messages.append(msg_entry)
        save_json(agent_messages_path, messages)

    def get_agent_conversation_context(self, email: str, agent_name: str) -> List[Dict]:
        """Get conversation context specific to an agent"""
        folder = get_session_folder(email)
        agent_messages_path = os.path.join(folder, f"{agent_name}_messages.json")
        
        if os.path.exists(agent_messages_path):
            messages = load_json(agent_messages_path)
            # Return last 5 messages for context
            return messages[-5:] if len(messages) > 5 else messages
        return []

    def add_user_message_to_agents(self, email: str, user_message: str, agents_to_run: List[str]):
        """Add user message to each agent's conversation context"""
        for agent_name in agents_to_run:
            self.save_agent_message(email, agent_name, user_message, role="user")

    def build_context_message(self, email: str, agent_name: str = None) -> str:
        """Build comprehensive context message for agents"""
        context = self.get_conversation_context(email)
        
        context_message = "=== CONVERSATION CONTEXT ===\n\n"
        
        # Add attachment details if available
        if context.get("attachment_details"):
            context_message += "=== ATTACHMENT DETAILS ===\n"
            attachment_details = context["attachment_details"]
            for key, value in attachment_details.items():
                context_message += f"{key}: {value}\n"
            context_message += "\n"
        
        # Add conversation history
        if context.get("conversation_history"):
            context_message += "=== CONVERSATION HISTORY ===\n"
            for message in context["conversation_history"]:
                role = message["role"].upper()
                content = message["content"]
                context_message += f"{role}: {content}\n"
            context_message += "\n"
        
        # Add agent-specific conversation context if available
        if agent_name:
            agent_context = self.get_agent_conversation_context(email, agent_name)
            if agent_context:
                context_message += f"=== {agent_name.upper()} CONVERSATION HISTORY ===\n"
                for message in agent_context:
                    role = message["role"].upper()
                    content = message["content"]
                    context_message += f"{role}: {content}\n"
                context_message += "\n"
        
        # Add agent-specific instruction
        if agent_name:
            context_message += f"=== INSTRUCTION FOR {agent_name.upper()} ===\n"
            context_message += f"Please analyze the above context and provide your specialist assessment.\n"
        
        return context_message

    def save_follow_up(self, email: str, agent_name: str, response: str):
        """Save agent response to follow_up.json if it's not JSON schema"""
        folder = get_session_folder(email)
        follow_up_path = os.path.join(folder, FOLLOW_UP_FILE)
        
        if os.path.exists(follow_up_path):
            follow_up_data = load_json(follow_up_path)
        else:
            follow_up_data = {
                "responses": [],
                "last_updated": time.time()
            }
        
        follow_up_entry = {
            "agent": agent_name,
            "response": response,
            "timestamp": time.time()
        }
        
        follow_up_data["responses"].append(follow_up_entry)
        follow_up_data["last_updated"] = time.time()
        save_json(follow_up_path, follow_up_data)

    def init_claim_state(self, email: str):
        """Initialize claim state"""
        folder = get_session_folder(email)
        claim_path = os.path.join(folder, CLAIM_FILE)
        
        if not os.path.exists(claim_path):
            default_claim = {
                "stage": "NEW",
                "incident_types": {},
                "completion_status": "in_progress",
                "agents_run": [],
                "agent_threads": {}  # Store thread IDs for each agent
            }
            save_json(claim_path, default_claim)

    def get_claim(self, email: str) -> Dict:
        """Get current claim state"""
        folder = get_session_folder(email)
        claim_path = os.path.join(folder, CLAIM_FILE)
        return load_json(claim_path) if os.path.exists(claim_path) else {}

    def save_agent_data(self, email: str, agent_name: str, data: Dict):
        """Save agent structured data to agent-specific data file"""
        folder = get_session_folder(email)
        agent_data_path = os.path.join(folder, f"{agent_name}_data.json")
        
        if os.path.exists(agent_data_path):
            existing_data = load_json(agent_data_path)
        else:
            existing_data = []
        
        data_entry = {
            "timestamp": time.time(),
            "data": data
        }
        existing_data.append(data_entry)
        save_json(agent_data_path, existing_data)

    def is_json_response(self, response: str) -> bool:
        """Check if response is valid JSON"""
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False

    def get_or_create_thread(self, email: str, agent_name: str) -> str:
        """Get existing thread ID or create new one for agent"""
        claim = self.get_claim(email)
        agent_threads = claim.get("agent_threads", {})
        
        if agent_name in agent_threads:
            return agent_threads[agent_name]
        
        # Create new thread
        thread = self.client.beta.threads.create()
        agent_threads[agent_name] = thread.id
        
        # Update claim with new thread ID
        claim["agent_threads"] = agent_threads
        save_claim_state(email, claim)
        
        return thread.id

    def run_assistant_agent(self, email: str, agent_name: str) -> bool:
        """Run assistant agent with comprehensive context"""
        if agent_name not in self.assistant_ids:
            print(f"[orchestration] No assistant ID found for agent: {agent_name}")
            return False
        
        try:
            assistant_id = self.assistant_ids[agent_name]
            thread_id = self.get_or_create_thread(email, agent_name)
            
            # Build comprehensive context message
            context_message = self.build_context_message(email, agent_name)
            
            # Save the context message as a user message to the agent's conversation history
            self.save_agent_message(email, agent_name, context_message, role="user")
            
            # Add message to thread
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=context_message
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            
            # Wait for completion and handle different statuses
            while run.status in ['queued', 'in_progress', 'cancelling']:
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
            
            # Handle requires_action status (when assistant tries to call functions)
            if run.status == 'requires_action':
                print(f"[orchestration] {agent_name} requires action - submitting empty tool outputs to continue")
                # Submit empty tool outputs to continue without function calls
                try:
                    self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=[]
                    )
                    # Wait for completion after submitting empty outputs
                    while run.status in ['queued', 'in_progress', 'cancelling']:
                        time.sleep(1)
                        run = self.client.beta.threads.runs.retrieve(
                            thread_id=thread_id,
                            run_id=run.id
                        )
                except Exception as tool_error:
                    print(f"[orchestration] Error handling tool outputs for {agent_name}: {tool_error}")
                    # Try to cancel the run and continue
                    try:
                        self.client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                    except:
                        pass
                    print(f"[orchestration] {agent_name} function call handling failed - continuing without response")
                    return False
            
            if run.status == 'completed':
                # Get the response
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread_id,
                    order="desc",
                    limit=1
                )
                
                if messages.data:
                    response_content = messages.data[0].content[0].text.value
                    
                    # Save assistant response to agent's conversation history
                    self.save_agent_message(email, agent_name, response_content, role="assistant")
                    
                    # Check if response is JSON
                    if self.is_json_response(response_content):
                        # Parse and save as structured data
                        try:
                            json_data = json.loads(response_content)
                            self.save_agent_data(email, agent_name, json_data)
                            print(f"[orchestration] {agent_name} returned structured data")
                        except json.JSONDecodeError:
                            # Fallback to follow-up storage
                            self.save_follow_up(email, agent_name, response_content)
                            print(f"[orchestration] {agent_name} returned message (JSON parse failed)")
                    else:
                        # Save as follow-up message
                        self.save_follow_up(email, agent_name, response_content)
                        print(f"[orchestration] {agent_name} returned conversational message")
                    
                    return True
                else:
                    print(f"[orchestration] No response from {agent_name}")
                    return False
            else:
                print(f"[orchestration] Assistant run failed with status: {run.status}")
                return False
                
        except Exception as e:
            print(f"[orchestration] Error running assistant agent {agent_name}: {e}")
            return False
    def get_agents_to_run(self, email: str) -> List[str]:
        """Get agents to run based on incident_types from triage"""
        claim = self.get_claim(email)
        incident_types = claim.get("incident_types", {})
        agents_already_run = claim.get("agents_run", [])
        
        agents_to_run = []
        
        # For each incident type that was identified, add corresponding agent
        for incident_type in incident_types:
            if incident_type in self.incident_type_to_agent:
                agent_name = self.incident_type_to_agent[incident_type]
                # if agent_name not in agents_already_run:
                # Need to get a seperate logic to figure out agents that have completed gathering information.
                agents_to_run.append(agent_name)
        
        return agents_to_run

    def run_agent(self, email: str, agent_name: str) -> bool:
        """Run a specific agent with comprehensive context"""
        print(f"[orchestration] Running agent: {agent_name}")
        
        try:
            # Special handling for triage agent
            if agent_name == "triage":
                context = self.get_conversation_context(email)
                # Convert context to the format expected by triage
                conversation_context = context.get("conversation_history", [])
                triage_result = run_triage(email, conversation_context)
                if triage_result:
                    claim = self.get_claim(email)
                    claim["incident_types"] = triage_result.get("incident_types", {})
                    claim["stage"] = "TRIAGED"
                    save_claim_state(email, claim)
                    print(f"[orchestration] Triage complete. Incident types: {claim['incident_types']}")
                    return True
                return False
            
            # Run assistant agent for all other agents
            else:
                success = self.run_assistant_agent(email, agent_name)
                
                if success:
                    # Mark agent as run
                    claim = self.get_claim(email)
                    agents_run = claim.get("agents_run", [])
                    if agent_name not in agents_run:
                        agents_run.append(agent_name)
                        claim["agents_run"] = agents_run
                        save_claim_state(email, claim)
                
                return success
                
        except Exception as e:
            print(f"[orchestration] Error running agent {agent_name}: {e}")
            return False

    def all_agents_complete(self, email: str) -> bool:
        """Check if all required agents based on incident_types are complete"""
        claim = self.get_claim(email)
        incident_types = claim.get("incident_types", {})
        agents_run = claim.get("agents_run", [])
        
        # Get all agents that should run based on incident types
        required_agents = []
        for incident_type in incident_types:
            if incident_type in self.incident_type_to_agent:
                required_agents.append(self.incident_type_to_agent[incident_type])
        
        # Check if all required agents have been run
        return all(agent in agents_run for agent in required_agents)

    def orchestrate(self, email: str, user_message: str, attachments: List[str]):
        print(f"\n[orchestrate] New message from {email}")
        self.init_claim_state(email)
        self.init_context(email)

        # Update conversation context with new message and attachments
        self.update_context(email, user_message, attachments)

        # Handle attachments
        if attachments:
            print("[orchestrate] Running attachment details agent...")
            generate_attachment_details(email, attachments)
            # Update context again to include attachment details
            self.update_context(email, "", [])  # Empty message, no new attachments

        claim = self.get_claim(email)
        stage = claim.get("stage", "NEW")
        print(f"[orchestrate] Current claim stage: {stage}")

        # Stage-specific handling
        if stage == "NEW":
            print("[orchestrate] First message - running clarifying question agent...")
            prelim = run_clarifying_question(email, user_message)
            
            claim["stage"] = "QUESTIONED"
            save_claim_state(email, claim)

        else:
            print("[orchestrate] Running triage on new message...")
            triage_success = self.run_agent(email, "triage")

            if triage_success:
                agents_to_run = self.get_agents_to_run(email)
                if agents_to_run:
                    print(f"[orchestrate] Running agents: {agents_to_run}")
                    # Add user message to each agent's conversation history before running
                    self.add_user_message_to_agents(email, user_message, agents_to_run)
                    
                    for agent_name in agents_to_run:
                        self.run_agent(email, agent_name)

                if self.all_agents_complete(email):
                    claim["stage"] = "COMPLETE"
                    save_claim_state(email, claim)
                    print("[orchestrate] All agents complete - claim processing finished.")

        print("[orchestrate] Orchestration complete.")


# Create global orchestrator instance
orchestrator = Orchestrator()

# Export the main function for backward compatibility
def orchestrate(email: str, user_message: str, attachments: List[str]):
    """Backward compatible orchestrate function"""
    return orchestrator.orchestrate(email, user_message, attachments)