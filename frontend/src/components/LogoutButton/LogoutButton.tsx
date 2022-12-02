import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ReactSession } from 'react-client-session';

import { BsPower } from 'react-icons/bs';
import './LogoutButton.scss';

const LogoutButton = () => {
    const navigate = useNavigate();

	const logout = () => {
		ReactSession.setStoreType("sessionStorage");
		ReactSession.set("USER_EMAIL", "");
		ReactSession.set("USER_NAME", "");
		ReactSession.set("USER_SURNAME", "");
		ReactSession.set("USER_ROLE", "");
		ReactSession.set("USER_ID", "");
		ReactSession.set("USER_COST", "");
		ReactSession.set("USER_CF", "");
		navigate('/login');
	}

    return (
		<div className="logoutbutton_container">
			<button onClick={() => logout()}><BsPower /></button>
		</div>
	)
}

export default LogoutButton