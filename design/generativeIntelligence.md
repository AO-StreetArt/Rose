# Generative 3D Intelligence

How do we go from "I know things exist around me and can respond to stimuli", to "What if this thing existed in my environment?"?

Well, we can actually leverage generative AI image systems here, like Sora.  But, we can bypass the prompt from the user, generating it based on tags from related features we know about.

## Finding Related Features

Looking at our feature storage by tag, we want to look for related tags amongst the features, and then aggregate the tags of all *those*.  We can then use this to create a prompt, and leverage genAI to suggest what might go here.  It's not recall, or prediction - it's creativity.  It's suggesting what sort of couch might go nice in a living room, based on what's there.