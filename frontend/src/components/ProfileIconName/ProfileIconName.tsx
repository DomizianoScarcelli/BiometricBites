import React from "react"

import "./ProfileIconName.scss"

type ProfileIconNameProps = {
	name: string
}

const ProfileIconName = ({ name }: ProfileIconNameProps) => {
	return (
		<div className="container">
			<div className="profileIcon">
				<p>{name[0]}</p>
			</div>
			<p>
				Good Evening, <br /> {name}!
			</p>
		</div>
	)
}

export default ProfileIconName
