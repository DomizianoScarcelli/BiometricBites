import axios from "axios";
import React, { useEffect, useState } from "react";
import { ReactSession } from 'react-client-session';

import "./LoginPage.scss";

function LoginPage() {
    const [formValues, setFormValues] = useState({ username: '', password: '' });
    const [loadingButton, setLoadingButton] = useState(false);
    const [popup, setPopup] = useState({trigger: false, title: '', description: ''});

    const handleSubmit = async (e: any) => {
        e.preventDefault();
        setLoadingButton(true);

        let formData = {
            username: formValues['username'],
            password: formValues['password']
        }

        axios.post('', formData)
        .then(function(response) {
            setLoadingButton(false);
        })
        .catch(function(error) {
            setLoadingButton(false);
        })
    }

    const onChange = (e: any) => {
        setFormValues({...formValues, [e.target.name]: e.target.value})
    }

	return (
		<div className="background">
			<div className="centerContainer">
                <div className="loginContainer">
                    <p>Login</p>
                    <form className='loginForm' onSubmit={handleSubmit}>
                        <input id='email' name='email' type='text' value={formValues['username']} onChange={onChange} placeholder='Email' />
                        <input id='password' name='password' type='password' value={formValues['password']} onChange={onChange} placeholder='Password' />
                        <button type='submit'>
                            {loadingButton ? 'Signing in..' : 'Sign in'}
                        </button>
                    </form>
                </div>
            </div>
		</div>
	)
}

export default LoginPage
