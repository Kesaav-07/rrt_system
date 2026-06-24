"""
auth.py
-------
Authentication module for the RRT BiLSTM Surveillance Dashboard.

Features:
    - SHA-256 salted password hashing (no plaintext stored)
    - Roles: Nurse, Physician, RRT Team, Admin
    - Signup / Login / Logout
    - Role stored permanently in users.csv at signup
    - Role changes only allowed by Admin through Admin panel
    - Session management via st.session_state

Storage:
    users.csv with columns: username, salt, password_hash, role, created_at
"""

from __future__ import annotations

import os
import sys
import hashlib
import secrets
import pandas as pd
from config import USERS_FILE, ROLES, DATA_DIR
from db import get_connection
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import USERS_FILE, ROLES, DATA_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _hash_password(password: str, salt: str) -> str:
    """SHA-256 hash of salt+password."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _load_users() -> pd.DataFrame:
    try:
        conn = get_connection()
        df = pd.read_sql("SELECT username, salt, password_hash, role, created_at FROM users", conn)
        conn.close()
        return df.fillna("")
    except Exception as exc:
        st.error(f"Failed to load users from MySQL: {exc}")
        return pd.DataFrame(
            columns=["username", "salt", "password_hash", "role", "created_at"]
        )


def _save_users(df: pd.DataFrame) -> None:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users")
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO users
                (username, salt, password_hash, role, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                row["username"],
                row["salt"],
                row["password_hash"],
                row["role"],
                row["created_at"],
            ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as exc:
        st.error(f"Failed to save users to MySQL: {exc}")


# ---------------------------------------------------------------------------
# User management (called from Admin panel)
# ---------------------------------------------------------------------------

def create_user(username: str, password: str, role: str) -> tuple[bool, str]:
    """
    Create a new user account.

    Args:
        username: Unique username.
        password: Plaintext password (will be hashed).
        role:     One of ROLES.

    Returns:
        (success, message)
    """
    users = _load_users()
    if username in users["username"].values:
        return False, "That username is already taken."
    if role not in ROLES:
        return False, f"Invalid role. Choose from: {', '.join(ROLES)}"

    salt = secrets.token_hex(8)
    pw_hash = _hash_password(password, salt)
    new_row = pd.DataFrame([{
        "username":      username,
        "salt":          salt,
        "password_hash": pw_hash,
        "role":          role,
        "created_at":    datetime.now().isoformat(timespec="seconds"),
    }])
    users = pd.concat([users, new_row], ignore_index=True)
    _save_users(users)
    return True, "Account created — you can log in now."


def verify_login(username: str, password: str) -> tuple[bool, str | None, str]:
    """
    Verify login credentials.

    Args:
        username: Submitted username.
        password: Submitted plaintext password.

    Returns:
        (success, role_or_None, message)
    """
    users = _load_users()
    match = users[users["username"] == username]
    if match.empty:
        return False, None, "No account found with that username."
    row = match.iloc[0]
    if _hash_password(password, row["salt"]) != row["password_hash"]:
        return False, None, "Incorrect password."
    return True, row["role"], "Login successful."


def change_user_role(username: str, new_role: str) -> tuple[bool, str]:
    """
    Change a user's role. Only callable from the Admin panel.

    Args:
        username: Target username.
        new_role: New role from ROLES.

    Returns:
        (success, message)
    """
    if new_role not in ROLES:
        return False, f"Invalid role: {new_role}"
    users = _load_users()
    idx = users[users["username"] == username].index
    if idx.empty:
        return False, f"User '{username}' not found."
    users.loc[idx, "role"] = new_role
    _save_users(users)
    return True, f"Role updated to '{new_role}' for {username}."


def delete_user(username: str) -> tuple[bool, str]:
    """
    Delete a user account. Admin-only operation.

    Args:
        username: Username to delete.

    Returns:
        (success, message)
    """
    users = _load_users()
    if username not in users["username"].values:
        return False, f"User '{username}' not found."
    users = users[users["username"] != username].reset_index(drop=True)
    _save_users(users)
    return True, f"User '{username}' deleted."


def list_users() -> pd.DataFrame:
    """Return DataFrame of all users (without salt and password_hash)."""
    users = _load_users()
    return users[["username", "role", "created_at"]].copy()


# ---------------------------------------------------------------------------
# Streamlit UI — login form
# ---------------------------------------------------------------------------

def _login_form() -> None:
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log In", use_container_width=True)

    if submitted:
        if not username.strip() or not password:
            st.error("Please enter your username and password.")
            return
        ok, role, msg = verify_login(username.strip(), password)
        if ok:
            st.session_state["authenticated"] = True
            st.session_state["username"]       = username.strip()
            st.session_state["user_role"]      = role
            st.session_state["view"]           = "dashboard"
            st.session_state["selected_patient"] = None
            st.rerun()
        else:
            st.error(msg)


# ---------------------------------------------------------------------------
# Streamlit UI — signup form
# ---------------------------------------------------------------------------

def _signup_form() -> None:
    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Choose a username", key="signup_username")
        password = st.text_input("Choose a password", type="password", key="signup_pw")
        confirm  = st.text_input("Confirm password",  type="password", key="signup_confirm")
        role     = st.selectbox("Role", ROLES, key="signup_role")
        submitted = st.form_submit_button("Create Account", use_container_width=True)

    if submitted:
        if not username.strip() or not password:
            st.error("Username and password are required.")
        elif password != confirm:
            st.error("Passwords do not match.")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            ok, msg = create_user(username.strip(), password, role)
            if ok:
                st.success(f"{msg} Switch to the Log In tab.")
            else:
                st.error(msg)


# ---------------------------------------------------------------------------
# Streamlit UI — auth gate (call at top of app.py)
# ---------------------------------------------------------------------------

def render_auth_gate() -> None:
    """
    Render the login/signup screen and halt with st.stop() until the user
    is authenticated.

    Once authenticated, sets in st.session_state:
        authenticated : True
        username      : str
        user_role     : str (one of ROLES)
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return  # Already logged in — continue

    # ---- Auth page styling ----
    st.markdown("""
    <style>
    :root, .stApp { color-scheme: light !important; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"], body { background-color: #F4F6F8 !important; }
    div[data-testid="stMarkdownContainer"] p { color: #0B2545 !important; }
    h1, h2, h3 { color: #0B2545 !important; }
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #0B2545 !important;
        border: 1px solid #DCE3E8 !important;
    }
    .stTextInput label p { color: #0B2545 !important; }
    .stTextInput button { background-color: #FFFFFF !important; }
    .stTextInput button svg { fill: #5A7184 !important; }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0B2545 !important;
        border: 1px solid #DCE3E8 !important;
    }
    div[data-baseweb="select"] * { color: #0B2545 !important; }
    ul[data-testid="stSelectboxVirtualDropdown"] li,
    div[data-baseweb="popover"] li {
        background-color: #FFFFFF !important;
        color: #0B2545 !important;
    }
    button[data-baseweb="tab"] p { color: #0B2545 !important; }
    div[data-baseweb="tab-list"] { border-bottom: 1px solid #DCE3E8 !important; }
    div[data-baseweb="tab-highlight"] { background-color: #0F6E68 !important; }
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE3E8 !important;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .stButton button, button[kind="formSubmit"], button[kind="secondaryFormSubmit"] {
        background-color: #0F6E68 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    </style>
    <div style="max-width:500px; margin:3rem auto 1.5rem auto; text-align:center;">
        <div style="font-size:3rem;">🩺</div>
        <h1 style="margin:0.3rem 0 0.5rem 0;">RRT BiLSTM Dashboard</h1>
        <p style="color:#5A7184; margin:0;">
            Real-Time Patient Deterioration Forecasting &amp; RRT Surveillance
        </p>
    </div>
    """, unsafe_allow_html=True)

    center = st.columns([1, 1.6, 1])[1]
    with center:
        tab_login, tab_signup = st.tabs(["🔑 Log In", "📝 Sign Up"])
        with tab_login:
            _login_form()
        with tab_signup:
            _signup_form()

    st.stop()


# ---------------------------------------------------------------------------
# Streamlit UI — logout control (call inside sidebar after login)
# ---------------------------------------------------------------------------

def render_logout_control() -> None:
    """Render the user info + logout button in the sidebar."""
    username  = st.session_state.get("username", "")
    user_role = st.session_state.get("user_role", "")

    st.sidebar.markdown(
        f"""
        <div style="background:#112F58; border-radius:8px; padding:0.6rem 0.8rem;
                    border:1px solid #1C4373; margin-bottom:0.5rem;">
            <div style="font-size:0.72rem; color:#8FD9CE; letter-spacing:0.06em;
                        text-transform:uppercase;">Signed in as</div>
            <div style="font-size:0.92rem; color:#FFFFFF; font-weight:600;">{username}</div>
            <div style="font-size:0.78rem; color:#B9C7DA;">Role: {user_role}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        for key in ("authenticated", "username", "user_role",
                    "view", "selected_patient"):
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------------------------
# Streamlit UI — Admin panel (user management)
# ---------------------------------------------------------------------------

def render_admin_panel() -> None:
    """
    Render the Admin user-management panel.
    Only accessible when st.session_state['user_role'] == 'Admin'.
    """
    if st.session_state.get("user_role") != "Admin":
        st.error("Admin access only.")
        return

    st.subheader("👤 User Management")

    users_df = list_users()

    st.markdown("##### All Users")
    st.dataframe(users_df, use_container_width=True, height=250)

    st.markdown("---")
    col_change, col_delete = st.columns(2)

    with col_change:
        st.markdown("##### Change Role")
        with st.form("change_role_form"):
            target_user = st.selectbox(
                "Select user",
                options=users_df["username"].tolist(),
                key="admin_target_user",
            )
            new_role = st.selectbox("New role", ROLES, key="admin_new_role")
            if st.form_submit_button("Update Role", use_container_width=True):
                ok, msg = change_user_role(target_user, new_role)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with col_delete:
        st.markdown("##### Delete User")
        with st.form("delete_user_form"):
            del_user = st.selectbox(
                "Select user to delete",
                options=users_df["username"].tolist(),
                key="admin_del_user",
            )
            confirm = st.checkbox(
                "I confirm this user should be permanently deleted.",
                key="admin_del_confirm",
            )
            if st.form_submit_button("Delete User", use_container_width=True):
                if not confirm:
                    st.warning("Please check the confirmation box.")
                elif del_user == st.session_state.get("username"):
                    st.error("You cannot delete your own account.")
                else:
                    ok, msg = delete_user(del_user)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    st.markdown("---")
    st.markdown("##### Add New User")
    with st.form("admin_add_user_form"):
        new_username = st.text_input("Username", key="admin_new_username")
        new_password = st.text_input("Password", type="password", key="admin_new_pw")
        new_role_add = st.selectbox("Role", ROLES, key="admin_new_role_add")
        if st.form_submit_button("Create User", use_container_width=True):
            if not new_username.strip() or not new_password:
                st.error("Username and password required.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                ok, msg = create_user(new_username.strip(), new_password, new_role_add)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
