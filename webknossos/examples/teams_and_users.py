import webknossos as wk


def main() -> None:
    # Get the current user
    current_user = wk.User.get_current_user()
    print(
        f"You are currently logged in as: {current_user.first_name} {current_user.last_name} ({current_user.email})."
    )

    # Get all users managed by the current user
    all_my_users = wk.User.get_all_managed_users()
    print("Managed users:")
    for user in all_my_users:
        print(f"\t{user.first_name} {user.last_name} ({user.email})")

    # Get teams of current user
    all_my_teams = wk.Team.get_list()
    print("Teams:")
    for team in all_my_teams:
        print(f"\t{team.name} ({team.organization_id})")

    # Add a new team
    wk.Team.add("My new team")

    # Set current user as team manager
    current_user.assign_team_roles("My new team", is_team_manager=True)


if __name__ == "__main__":
    main()
