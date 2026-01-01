from agent import StudyCopilotAgent
import re

def parse_command(user_input: str) -> tuple:
    # Command parsing
    input_lower = user_input.lower()
    
    if 'quiz' in input_lower:
        # calls quiz genertator tool
        topic = re.sub(r'quiz\s*(me\s+on\s*)?', '', input_lower, flags=re.IGNORECASE)
        return ('quiz', topic.strip(), 5)
    
    elif 'revise' in input_lower or 'study' in input_lower:
        # calls revision planner
        match = re.search(r'(?:revise|study)\s+([^\d]+?)\s*(\d+)', input_lower)
        if match:
            topic, minutes = match.groups()
            return ('revise', topic.strip(), int(minutes))
        return ('revise', input_lower.replace('revise', '').replace('study', '').strip(), 60)
    
    else:
        #runs a general query
        return ('ask', user_input, None)

def main():
    print("🤖:Personal Study Copilot Agent")
    print("Commands: 'quiz <topic>', 'revise <topic> <minutes>', or ask questions")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    agent = StudyCopilotAgent()
    
    while True:
        user_input = input("\n❓ You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        # Agent plans: parse → decide tool → execute
        action, param1, param2 = parse_command(user_input)
        
        print(f"\n🧠 Agent plans: {action} '{param1}'")
        print("-" * 40)
        
        if action == 'quiz':
            response = agent.generate_quiz(param1, param2 or 5)
        elif action == 'revise':
            response = agent.revision_plan(param1, param2 or 60)
        else:
            response = agent.ask(param1)
        
        print("🤖 Agent:", response, "\n")

if __name__ == "__main__":
    main()
