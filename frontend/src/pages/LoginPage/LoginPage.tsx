import React, { useState, useEffect } from "react";
import { useNavigate } from 'react-router-dom';
import { ReactSession } from 'react-client-session';
import axios from "axios";

import { Popup } from '../../components';
import "./LoginPage.scss";

function LoginPage() {
    const [formValues, setFormValues] = useState({ email: '', password: '' });
    const [loadingButton, setLoadingButton] = useState(false);
    const [loggedIn, setLoggedIn] = useState(false);
    const [popup, setPopup] = useState({trigger: false, title: '', description: ''});
    const navigate = useNavigate();

    useEffect(() => {
        if (loggedIn || (ReactSession.get("USER_EMAIL") && ReactSession.get("USER_ROLE"))) {
            return navigate('/');
        }
    }, [loggedIn, navigate])

    const handleSubmit = async (e: any) => {
        e.preventDefault();
        setLoadingButton(true);

        let formData = new FormData();
        formData.append('email', formValues['email']);
        formData.append('password', formValues['password'])

        axios.post('http://localhost:8000/api/login', formData)
        .then(function(response) {
            console.log(response)
            if (response.status === 200) {
                let data = JSON.parse(response.data.data);
                ReactSession.setStoreType("sessionStorage");
                ReactSession.set("USER_EMAIL", data['EMAIL']);
                ReactSession.set("USER_NAME", data['NAME']);
                ReactSession.set("USER_SURNAME", data['SURNAME']);
                ReactSession.set("USER_ROLE", data['ROLE']);
                ReactSession.set("USER_ID", data['ID']);
                ReactSession.set("USER_COST", data['COST']);
                ReactSession.set("USER_CF", data['CF']);
                setLoadingButton(false);
                setLoggedIn(true)
            } else {
                setPopup({'trigger': true, 'title': 'An error occurred!', 'description': 'Please try again.'});
                setLoadingButton(false);
            }
        })
        .catch(function(error) {
            setPopup({'trigger': true, 'title': 'An error occurred!', 'description': error.response.data.message});
            setLoadingButton(false);
        })
    }

    const onChange = (e: any) => {
        setFormValues({...formValues, [e.target.name]: e.target.value})
    }

    const closePopup = () => {
        setPopup({...popup, 'trigger': false});
    }

	return (
        <>
            <div className="background">
                <div className="centerContainer">
                    <div className="loginContainer">
                        <p>Login</p>
                        <form className='loginForm' onSubmit={handleSubmit}>
                            <input id='email' name='email' type='text' value={formValues['email']} onChange={onChange} placeholder='Email' required />
                            <input id='password' name='password' type='password' value={formValues['password']} onChange={onChange} placeholder='Password' required />
                            <button type='submit'>
                                {loadingButton ? 'Signing in..' : 'Sign in'}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
            <Popup trigger={popup['trigger']} title={popup['title']} description={popup['description']} onClick={closePopup} />
        </>
	)
}

export default LoginPage
