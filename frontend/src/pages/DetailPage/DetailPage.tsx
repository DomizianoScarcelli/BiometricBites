import React, { useEffect } from 'react';
import { ReactSession } from 'react-client-session';
import { useNavigate } from 'react-router-dom';

import { BackButton, ProfileIconName } from '../../components';
import { AiOutlineUser, AiOutlineEuroCircle, AiOutlineMail, AiOutlineInfoCircle } from 'react-icons/ai';
import './DetailPage.scss';

function DetailPage() {
    const navigate = useNavigate();

    const firstLetterUppercase = (str: string) => {
		return str.charAt(0).toUpperCase() + str.slice(1);
	}

    useEffect (() => {
        ReactSession.setStoreType("sessionStorage");
		if (ReactSession.get("USER_EMAIL") === undefined)
		{
			navigate('/login');
		}
        if (ReactSession.get("USER_ROLE") === "admin")
        {
            navigate('/');
        }
	}, [])
	
	return (
        <div className='background'>
            <ProfileIconName name={ReactSession.get("USER_EMAIL") !== undefined ? firstLetterUppercase(ReactSession.get("USER_NAME"))+" "+firstLetterUppercase(ReactSession.get("USER_SURNAME")) : ''} />
            <BackButton link='/' />
            <div className='centralContainer'>
                <div className='detailContainer'>
                    <div className='detailContainerText'>
                        <p>Your Details</p>
                    </div>
                    <div className='detailContainerItems'>
                        <div className='detailItem'>
                            <span><strong><AiOutlineUser />  Name/Surname</strong></span>
                            <span>{firstLetterUppercase(ReactSession.get("USER_NAME"))+" "+firstLetterUppercase(ReactSession.get("USER_SURNAME"))}</span>
                        </div>
                        <div className='detailItem'>
                            <span><strong><AiOutlineMail />  Email:</strong></span>
                            <span>{ReactSession.get("USER_EMAIL")}</span>
                        </div>
                        <div className='detailItem'>
                            <span><strong><AiOutlineInfoCircle />  Fiscal Code:</strong></span>
                            <span>{ReactSession.get("USER_CF")}</span>
                        </div>
                        <div className='detailItem'>
                            <span><strong><AiOutlineEuroCircle />  You Pay:</strong></span>
                            <span>{'€'+ReactSession.get("USER_COST")}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
	)
}

export default DetailPage