import React from 'react';
import { useNavigate } from 'react-router-dom';

import { MdArrowBack } from 'react-icons/md';
import './BackButton.scss';

type BackButtonProps = {
	link: string
}

const BackButton = ({ link }: BackButtonProps) => {
    const navigate = useNavigate();

    return (
		<div className="backbutton_container">
			<button onClick={() => navigate(link)}><MdArrowBack /></button>
		</div>
	)
}

export default BackButton