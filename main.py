from agent.core import create_initial_messages, run_agent


def main() -> None:
    messages = create_initial_messages()
    context = {
        "user_role": "student",
        "page": "general"
    }

    while True:
        user_input = input("\n请输入你的问题，输入 exit 退出：")

        if user_input == "exit":
            print("已退出。")
            break

        answer = run_agent(messages, user_input, debug=True, context=context)
        print("\n最终回答：")
        print(answer)


if __name__ == "__main__":
    main()
